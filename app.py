import base64
import email
import imaplib
import json
import os
import re
import uuid
from datetime import date, datetime, time, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import streamlit as st
from openai import OpenAI

import fitz  # PyMuPDF
from audit_log import AuditLogger, redact_payload
from graph_client import GraphClient, GraphClientError
from ics_utils import build_ics_invite


st.set_page_config(
    page_title="PowerDash Interview Scheduler",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1400px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
#  CONFIG HELPERS
# =========================

def get_secret(key: str, default: str = "") -> str:
    return st.secrets.get(key, st.secrets.get(key.lower(), os.environ.get(key, default)))


def build_timezone_options() -> List[str]:
    common = [
        "UTC",
        "Europe/London",
        "Europe/Paris",
        "Europe/Berlin",
        "Europe/Dublin",
        "America/New_York",
        "America/Chicago",
        "America/Denver",
        "America/Los_Angeles",
        "America/Toronto",
        "Asia/Singapore",
        "Asia/Hong_Kong",
        "Asia/Tokyo",
        "Australia/Sydney",
    ]
    return common


# Secrets / env config
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", get_secret("openai_api_key", ""))
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

GRAPH_TENANT_ID = get_secret("graph_tenant_id")
GRAPH_CLIENT_ID = get_secret("graph_client_id")
GRAPH_CLIENT_SECRET = get_secret("graph_client_secret")
GRAPH_SCHEDULER = get_secret("graph_scheduler_mailbox", "scheduling@powerdashhr.com")

SMTP_USER = get_secret("smtp_username", get_secret("SMTP_USER", ""))
SMTP_PASSWORD = get_secret("smtp_password", get_secret("SMTP_PASSWORD", ""))
SMTP_HOST = get_secret("smtp_host", get_secret("SMTP_HOST", "smtp.gmail.com"))
SMTP_PORT = int(get_secret("smtp_port", str(get_secret("SMTP_PORT", "587"))))
SMTP_FROM = get_secret("smtp_from", SMTP_USER)

IMAP_HOST = get_secret("IMAP_HOST", "imap.gmail.com")
IMAP_PORT = int(get_secret("IMAP_PORT", "993"))

DEFAULT_TZ = get_secret("default_timezone", "UTC")
AUDIT_PATH = get_secret("audit_log_path", "audit_log.db")

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY is not set in Streamlit secrets or environment.")

client = OpenAI()
audit_logger = AuditLogger(AUDIT_PATH)
graph_client = GraphClient(
    tenant_id=GRAPH_TENANT_ID,
    client_id=GRAPH_CLIENT_ID,
    client_secret=GRAPH_CLIENT_SECRET,
    scheduler_mailbox=GRAPH_SCHEDULER,
)

# Session state
st.session_state.setdefault("slots", [])
st.session_state.setdefault("email_body", "")
st.session_state.setdefault("parsed_replies", [])
st.session_state.setdefault("latest_event_id", None)
st.session_state.setdefault("latest_ics", None)


# =========================
#  EMAIL HELPERS
# =========================

def send_plain_email(to_email: str, subject: str, body: str, cc: Optional[list[str]] = None) -> None:
    import smtplib

    if not SMTP_USER or not SMTP_PASSWORD:
        raise RuntimeError("SMTP credentials are not configured.")

    msg = MIMEMultipart()
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"] = to_email
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    recipients = [to_email] + (cc or [])

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(msg["From"], recipients, msg.as_string())


def send_email_with_ics(
    to_emails: list[str],
    subject: str,
    body: str,
    ics_text: str,
    cc_emails: Optional[list[str]] = None,
) -> None:
    import smtplib

    if not SMTP_USER or not SMTP_PASSWORD:
        raise RuntimeError("SMTP credentials are not configured.")

    msg = MIMEMultipart("mixed")
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"] = ", ".join(to_emails)
    if cc_emails:
        msg["Cc"] = ", ".join(cc_emails)
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    ics_part = MIMEText(ics_text, "calendar;method=REQUEST")
    ics_part.add_header("Content-Disposition", "attachment", filename="interview_invite.ics")
    msg.attach(ics_part)

    recipients = to_emails + (cc_emails or [])
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(msg["From"], recipients, msg.as_string())


# =========================
#  PDF → PNG HELPER
# =========================

def pdf_to_png(file_bytes: bytes) -> bytes:
    try:
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        page = pdf.load_page(0)
        pix = page.get_pixmap(dpi=200)
        png_bytes = pix.tobytes("png")
        return png_bytes
    except Exception as e:  # pragma: no cover - visualization helper
        raise RuntimeError(f"PDF conversion failed: {e}")


# =========================
#  CALENDAR PARSING (CHAT + VISION)
# =========================

def parse_calendar(file_bytes: bytes, filename: str):
    if filename.lower().endswith(".pdf"):
        try:
            file_bytes = pdf_to_png(file_bytes)
            mime = "image/png"
        except Exception as e:
            st.error(str(e))
            return []
    else:
        mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"

    b64 = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    prompt = """
You are helping an in-house recruiter.

From this weekly free/busy calendar image, extract ALL 30-minute or longer FREE time blocks
that are suitable for interviews.

Return ONLY STRICT JSON in this exact shape (no comments, no extra keys, no text before/after):

{
  "slots": [
    {"date": "2025-11-30", "start": "09:00", "end": "09:30"},
    {"date": "2025-11-30", "start": "10:00", "end": "11:00"}
  ]
}

Rules:
- date format: YYYY-MM-DD
- time format: HH:MM (24-hour)
- ensure start < end
- only include times the calendar shows as FREE.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:  # pragma: no cover - API path
        st.error(f"Error calling OpenAI for calendar parsing: {e}")
        return []

    try:
        obj = json.loads(raw)
        if "slots" in obj and isinstance(obj["slots"], list):
            return obj["slots"]
        st.error("Model returned JSON, but missing valid 'slots' list.")
        st.code(raw)
        return []
    except Exception as e:
        st.error(f"Could not parse model JSON: {e}")
        st.code(raw)
        return []


# =========================
#  EMAIL GENERATION (CHAT)
# =========================

def generate_scheduling_email(
    cand_name: str,
    cand_email: str,
    hm_name: str,
    company: str,
    role: str,
    cand_tz: str,
    slots: list[dict],
):
    if not slots:
        return "No slots available."

    slot_lines = []
    for i, s in enumerate(slots, start=1):
        slot_lines.append(f"{i}. {s['date']} {s['start']}–{s['end']} ({cand_tz})")
    slot_text = "\n".join(slot_lines)

    prompt = f"""
You are an expert in-house recruiter.

Write a warm, concise and professional email to a job candidate to offer interview time options.

Details:
- Candidate: {cand_name} <{cand_email}>
- Hiring manager: {hm_name}
- Company: {company}
- Role: {role}
- Candidate time zone: {cand_tz}

Time options (already converted to candidate's timezone):
{slot_text}

Instructions:
- Clearly label the options with the numbers 1, 2, 3, ...
- Ask the candidate to reply ONLY with the option number that suits them best,
  or to propose alternative times if none work.
- Be friendly but businesslike.
- Sign off as the recruiter on behalf of the hiring manager.
- Do NOT include a subject line (body only).
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "You write clear, friendly, professional recruitment emails.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return resp.choices[0].message.content.strip()


# =========================
#  IMAP: CHECK SCHEDULER INBOX
# =========================

def check_scheduler_inbox(limit: int = 10):
    results = []
    if not SMTP_USER or not SMTP_PASSWORD:
        st.error("SMTP credentials not set – cannot check inbox.")
        return results

    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        mail.login(SMTP_USER, SMTP_PASSWORD)
        mail.select("INBOX")
        typ, data = mail.search(None, "UNSEEN")
        if typ != "OK":
            return results

        ids = data[0].split()
        if not ids:
            return results

        for msg_id in ids[-limit:]:
            typ, msg_data = mail.fetch(msg_id, "(RFC822)")
            if typ != "OK":
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            from_addr = email.utils.parseaddr(msg.get("From", ""))[1]
            subject = msg.get("Subject", "")
            body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    disp = str(part.get("Content-Disposition", ""))
                    if ctype == "text/plain" and "attachment" not in disp:
                        body_bytes = part.get_payload(decode=True) or b""
                        body = body_bytes.decode(errors="ignore")
                        break
            else:
                body_bytes = msg.get_payload(decode=True) or b""
                body = body_bytes.decode(errors="ignore")

            results.append(
                {
                    "from": from_addr,
                    "subject": subject,
                    "body": body,
                }
            )

        mail.logout()
    except Exception as e:  # pragma: no cover - external IMAP
        st.error(f"Error checking IMAP inbox: {e}")

    return results


def interpret_slot_choice(body: str, num_slots: int) -> int | None:
    numbers = re.findall(r"\b([1-9][0-9]?)\b", body)
    for n in numbers:
        val = int(n)
        if 1 <= val <= num_slots:
            return val
    return None


# =========================
#  TIME & GRAPH HELPERS
# =========================

def normalize_datetime(
    meeting_date: date,
    meeting_time: time,
    tz_name: str,
    duration_minutes: int,
):
    tz = ZoneInfo(tz_name)
    start_local = datetime.combine(meeting_date, meeting_time).replace(tzinfo=tz)
    end_local = start_local + timedelta(minutes=duration_minutes)
    return start_local, end_local, start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def build_graph_event_payload(
    subject: str,
    body: str,
    start_local: datetime,
    end_local: datetime,
    tz_name: str,
    candidate_email: str,
    hiring_manager_email: str,
    recruiter_email: Optional[str],
    include_recruiter: bool,
    interview_type: str,
    location_text: str,
    organizer_name: str,
    candidate_name: str,
    hiring_manager_name: str,
) -> Dict[str, Any]:
    attendees = [
        {
            "emailAddress": {"address": candidate_email, "name": candidate_name or candidate_email},
            "type": "required",
        },
        {
            "emailAddress": {"address": hiring_manager_email, "name": hiring_manager_name or hiring_manager_email},
            "type": "required",
        },
    ]
    if include_recruiter and recruiter_email:
        attendees.append(
            {
                "emailAddress": {"address": recruiter_email, "name": organizer_name or recruiter_email},
                "type": "optional",
            }
        )

    payload: Dict[str, Any] = {
        "subject": subject,
        "body": {"contentType": "HTML", "content": body.replace("\n", "<br>")},
        "start": {"dateTime": start_local.isoformat(), "timeZone": tz_name},
        "end": {"dateTime": end_local.isoformat(), "timeZone": tz_name},
        "attendees": attendees,
        "location": {"displayName": location_text},
    }

    if interview_type.lower().startswith("teams"):
        payload["isOnlineMeeting"] = True
        payload["onlineMeetingProvider"] = "teamsForBusiness"
    else:
        payload["isOnlineMeeting"] = False

    return payload


def build_ics_payload(
    start_dt_utc: datetime,
    end_dt_utc: datetime,
    subject: str,
    description: str,
    organizer_name: str,
    organizer_email: str,
    attendees: List[tuple[str, str]],
    location: str,
    uid: Optional[str] = None,
) -> str:
    ics_text = build_ics_invite(
        start_dt=start_dt_utc,
        end_dt=end_dt_utc,
        subject=subject,
        description=description,
        organizer_email=organizer_email,
        organizer_name=organizer_name,
        attendees=attendees,
        location=location,
        uid=uid,
    )
    candidate_email = attendees[0][1] if attendees else None
    audit_logger.log(
        action="ics_generated",
        status="success",
        actor=organizer_email,
        candidate_email=candidate_email,
        event_id=uid,
        payload={"subject": subject},
    )
    return ics_text


# =========================
#  UI LAYOUT
# =========================

st.title("PowerDash Interview Scheduler")
st.caption("Production-ready scheduling with PDF parsing, Microsoft Graph, and ICS fallback.")

tab_main, tab_inbox, tab_invites = st.tabs(
    ["New Scheduling Request", "Scheduler Inbox", "Calendar Invites"]
)


# -------------
# TAB 1: MAIN
# -------------
with tab_main:
    col_left, col_mid, col_right = st.columns([1.3, 1.0, 1.3])

    with col_left:
        st.subheader("Hiring Manager & Recruiter")

        hm_name = st.text_input("Hiring manager name", value="Martin McDonald")
        hm_email = st.text_input("Hiring manager email", value="martinmcd21@hotmail.com")
        hm_tz = st.text_input(
            "Hiring manager timezone (IANA, e.g. Europe/London, America/New_York)",
            value="Europe/London",
        )

        company = st.text_input("Company name", value="PowerDash HR")
        role_title = st.text_input("Role title", value="Architect")

        st.markdown("---")
        recruiter_name = st.text_input("Recruiter name", value="Amanda Hansen")
        recruiter_email = st.text_input("Recruiter email", value="info@powerdashhr.com")

    with col_mid:
        st.subheader("Upload Calendar (PDF/Image)")
        uploaded = st.file_uploader(
            "Upload PDF, PNG, JPG of hiring manager’s free/busy.",
            type=["pdf", "png", "jpg", "jpeg"],
        )

        parse_btn = st.button(
            "Parse availability",
            type="primary",
            disabled=uploaded is None,
        )

        if parse_btn and uploaded is not None:
            with st.spinner("Parsing calendar with GPT-4o-mini..."):
                slots = parse_calendar(uploaded.read(), uploaded.name)
                st.session_state["slots"] = slots

        slots = st.session_state.get("slots", [])

        if slots:
            st.markdown("**Detected free slots**")
            st.dataframe(slots, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Candidate")

        cand_name = st.text_input("Candidate name", value="Ruth Nicholson")
        cand_email = st.text_input("Candidate email", value="ruthnicholson1@hotmail.com")
        cand_tz = st.text_input(
            "Candidate timezone (IANA, e.g. Europe/London, America/New_York)",
            value="Europe/London",
        )

        st.markdown("### Scheduling email")

        gen_email_btn = st.button(
            "Generate scheduling email",
            disabled=not (cand_name and cand_email and cand_tz and slots),
        )

        if gen_email_btn and slots:
            with st.spinner("Generating email with GPT-4o-mini..."):
                body = generate_scheduling_email(
                    cand_name,
                    cand_email,
                    hm_name,
                    company,
                    role_title,
                    cand_tz,
                    slots,
                )
                st.session_state["email_body"] = body

        email_body = st.text_area(
            "Email preview (from scheduling mailbox)",
            value=st.session_state.get("email_body", ""),
            height=260,
        )

        if st.button(
            "Send email to candidate",
            disabled=not (email_body and cand_email and SMTP_USER),
        ):
            subject = f"Interview availability – {role_title} at {company}"
            try:
                send_plain_email(
                    cand_email,
                    subject,
                    email_body,
                    cc=[recruiter_email],
                )
                st.success("Email sent from scheduling mailbox to candidate. 🎉")
            except Exception as e:
                st.error(f"Error sending email: {e}")


# -------------
# TAB 2: INBOX
# -------------
with tab_inbox:
    st.subheader("Scheduler Inbox (unread replies)")

    st.write(
        "Click the button below to check the scheduling mailbox for unread replies. "
        "We will try to detect which option number the candidate has chosen."
    )

    check_btn = st.button("Check scheduler inbox now")

    if check_btn:
        with st.spinner("Checking IMAP inbox..."):
            messages = check_scheduler_inbox(limit=10)

        parsed_replies = []
        num_slots = len(st.session_state.get("slots", []))

        for msg in messages:
            chosen = interpret_slot_choice(msg["body"], num_slots)
            parsed_replies.append(
                {
                    "from": msg["from"],
                    "subject": msg["subject"],
                    "body": msg["body"],
                    "chosen_option": chosen,
                }
            )

        st.session_state["parsed_replies"] = parsed_replies

    parsed_replies = st.session_state.get("parsed_replies", [])

    if not parsed_replies:
        st.info("No parsed replies yet. Send a scheduling email and wait for replies.")
    else:
        st.success(f"Fetched and analysed {len(parsed_replies)} message(s).")

        for i, r in enumerate(parsed_replies, start=1):
            with st.expander(
                f"{i}. {r['subject']} — {r['from']} "
                + (
                    f"(chosen option: {r['chosen_option']})"
                    if r["chosen_option"]
                    else "(no clear option detected)"
                )
            ):
                st.text(r["body"])
                st.markdown(
                    f"**Detected option:** "
                    f"{r['chosen_option'] if r['chosen_option'] else 'Unclear'}"
                )


# -------------
# TAB 3: INVITES & GRAPH
# -------------
with tab_invites:
    st.subheader("Create & manage interview invites (Microsoft Graph)")

    graph_ready = graph_client.configured
    graph_status = "✅ Configured" if graph_ready else "⚠️ Configure graph_tenant_id, graph_client_id, graph_client_secret, graph_scheduler_mailbox in secrets."
    st.info(graph_status)
    tz_options = build_timezone_options()
    tz_default_index = tz_options.index(DEFAULT_TZ) if DEFAULT_TZ in tz_options else 0

    with st.expander("Graph diagnostics", expanded=False):
        st.write("Run quick checks against Microsoft Graph.")
        if st.button("Run diagnostics"):
            if not graph_ready:
                st.warning("Graph credentials are not configured.")
            else:
                with st.spinner("Running diagnostics..."):
                    results = graph_client.diagnostics()
                st.json(results)

    st.markdown("### New scheduling request (Graph)")
    with st.form("graph_create_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            candidate_email = st.text_input("Candidate email", value=st.session_state.get("candidate_email", ""))
            candidate_name = st.text_input("Candidate name", value="Candidate")
            include_recruiter = st.checkbox("Include recruiter as attendee", value=True)
        with col2:
            hiring_manager_email = st.text_input("Hiring manager email", value=hm_email)
            hiring_manager_name = st.text_input("Hiring manager name", value=hm_name)
            recruiter_email_field = st.text_input("Recruiter email", value=recruiter_email)
        with col3:
            meeting_date = st.date_input("Interview date", value=date.today())
            meeting_time = st.time_input("Start time", value=time(hour=10, minute=0))
            duration_minutes = st.number_input("Duration (minutes)", min_value=15, max_value=240, value=60, step=15)

        tz_choice = st.selectbox("Display timezone", options=tz_options, index=tz_default_index)
        subject = st.text_input("Invite subject", value=f"Interview – {role_title}")
        interview_type = st.radio("Interview type", options=["Teams", "On-site / phone"], horizontal=True)

        location_text = st.text_input(
            "Location or meeting info",
            value="Microsoft Teams" if interview_type == "Teams" else "Provide office location or dial-in",
        )
        invite_body = st.text_area(
            "Invitation message",
            value=(
                f"Hi all,<br><br>Interview for {role_title} at {company}.<br>"
                f"Candidate: {cand_name or 'Candidate'}<br>"
                f"Hiring manager: {hm_name or 'Hiring manager'}<br>"
                f"Recruiter: {recruiter_name or 'Recruiter'}<br><br>Best regards,<br>{recruiter_name or 'Recruiter'}"
            ),
            height=180,
        )

        submitted = st.form_submit_button("Create interview invite via Graph", disabled=not graph_ready)

    if submitted:
        if not (candidate_email and hiring_manager_email):
            st.error("Candidate and hiring manager emails are required.")
        else:
            try:
                start_local, end_local, start_utc, end_utc = normalize_datetime(
                    meeting_date, meeting_time, tz_choice, int(duration_minutes)
                )
                payload = build_graph_event_payload(
                    subject=subject,
                    body=invite_body,
                    start_local=start_local,
                    end_local=end_local,
                    tz_name=tz_choice,
                    candidate_email=candidate_email,
                    hiring_manager_email=hiring_manager_email,
                    recruiter_email=recruiter_email_field,
                    include_recruiter=include_recruiter,
                    interview_type=interview_type,
                    location_text=location_text,
                    organizer_name=recruiter_name,
                    candidate_name=candidate_name,
                    hiring_manager_name=hiring_manager_name,
                )
                result = graph_client.create_event(payload)
                event_id = result.get("id")
                st.session_state["latest_event_id"] = event_id
                audit_logger.log(
                    action="graph_create_event",
                    status="success",
                    actor=GRAPH_SCHEDULER,
                    candidate_email=candidate_email,
                    hiring_manager_email=hiring_manager_email,
                    recruiter_email=recruiter_email_field,
                    role_title=role_title,
                    event_id=event_id,
                    payload=redact_payload(payload, {"client_secret"}),
                )
                audit_logger.save_event(
                    event_id=event_id,
                    candidate_email=candidate_email,
                    hiring_manager_email=hiring_manager_email,
                    recruiter_email=recruiter_email_field,
                    role_title=role_title,
                    subject=subject,
                    start_utc=start_utc.isoformat(),
                    end_utc=end_utc.isoformat(),
                    timezone=tz_choice,
                    status="created",
                )
                ics_text = build_ics_payload(
                    start_dt_utc=start_utc,
                    end_dt_utc=end_utc,
                    subject=subject,
                    description=invite_body.replace("<br>", "\n"),
                    organizer_name=recruiter_name or "Scheduler",
                    organizer_email=SMTP_FROM or recruiter_email_field or GRAPH_SCHEDULER,
                    attendees=[
                        (candidate_name or "Candidate", candidate_email),
                        (hiring_manager_name or "Hiring manager", hiring_manager_email),
                    ],
                    location=location_text,
                    uid=event_id or str(uuid.uuid4()),
                )
                st.session_state["latest_ics"] = ics_text
                st.success(f"Graph event created and invites sent. Event ID: {event_id}")
            except GraphClientError as e:
                st.error(f"Graph create failed: {e} ({e.details})")
                audit_logger.log(
                    action="graph_create_failed",
                    status="failed",
                    actor=GRAPH_SCHEDULER,
                    candidate_email=candidate_email,
                    hiring_manager_email=hiring_manager_email,
                    recruiter_email=recruiter_email_field,
                    role_title=role_title,
                    payload={"status": e.status},
                    error_message=str(e),
                )
                ics_text = build_ics_payload(
                    start_dt_utc=start_utc,
                    end_dt_utc=end_utc,
                    subject=subject,
                    description=invite_body.replace("<br>", "\n"),
                    organizer_name=recruiter_name or "Scheduler",
                    organizer_email=SMTP_FROM or recruiter_email_field or GRAPH_SCHEDULER,
                    attendees=[
                        (candidate_name or "Candidate", candidate_email),
                        (hiring_manager_name or "Hiring manager", hiring_manager_email),
                    ],
                    location=location_text,
                    uid=str(uuid.uuid4()),
                )
                st.session_state["latest_ics"] = ics_text
                st.download_button(
                    "Download .ics fallback",
                    data=ics_text,
                    file_name="interview_invite.ics",
                    mime="text/calendar",
                    on_click=lambda: audit_logger.log(
                        action="ics_downloaded",
                        status="success",
                        actor=SMTP_FROM,
                        candidate_email=candidate_email,
                        hiring_manager_email=hiring_manager_email,
                        recruiter_email=recruiter_email_field,
                        role_title=role_title,
                    ),
                )
            except Exception as e:  # pragma: no cover - UI safety
                st.error(f"Unexpected error: {e}")

    st.markdown("---")
    st.markdown("### Reschedule existing event")
    stored_events = audit_logger.scheduled_events()
    event_options = [f"{ev['subject']} ({ev['event_id']})" for ev in stored_events]
    chosen_event_id: Optional[str] = None
    if event_options:
        selected_label = st.selectbox("Select event to reschedule", options=["-- select --"] + event_options)
        if selected_label != "-- select --":
            chosen_event_id = stored_events[event_options.index(selected_label)]["event_id"]
    manual_event_id = st.text_input("Or enter event ID", value=st.session_state.get("latest_event_id") or "")
    event_to_use = chosen_event_id or manual_event_id

    col_rs1, col_rs2 = st.columns(2)
    with col_rs1:
        new_date = st.date_input("New date", value=date.today())
        new_time = st.time_input("New start time", value=time(hour=11))
    with col_rs2:
        new_duration = st.number_input("New duration (minutes)", min_value=15, max_value=240, value=60, step=15)
        new_tz = st.selectbox("Timezone", options=tz_options, index=tz_default_index)

    if st.button("Reschedule via Graph", disabled=not (graph_ready and event_to_use)):
        try:
            start_local, end_local, start_utc, end_utc = normalize_datetime(new_date, new_time, new_tz, int(new_duration))
            patch_payload = {
                "start": {"dateTime": start_local.isoformat(), "timeZone": new_tz},
                "end": {"dateTime": end_local.isoformat(), "timeZone": new_tz},
            }
            graph_client.update_event(event_to_use, patch_payload)
            audit_logger.log(
                action="graph_reschedule_event",
                status="success",
                actor=GRAPH_SCHEDULER,
                event_id=event_to_use,
                payload=patch_payload,
            )
            audit_logger.save_event(
                event_id=event_to_use,
                candidate_email="",
                hiring_manager_email="",
                recruiter_email="",
                role_title=role_title,
                subject=subject,
                start_utc=start_utc.isoformat(),
                end_utc=end_utc.isoformat(),
                timezone=new_tz,
                status="rescheduled",
            )
            st.success("Event rescheduled. Teams link preserved.")
        except GraphClientError as e:
            st.error(f"Reschedule failed: {e} ({e.details})")
            audit_logger.log(
                action="graph_reschedule_failed",
                status="failed",
                actor=GRAPH_SCHEDULER,
                event_id=event_to_use,
                payload={"status": e.status},
                error_message=str(e),
            )

    st.markdown("---")
    st.markdown("### Cancel interview")
    cancel_event_id = st.text_input("Event ID to cancel", value=event_to_use)
    if st.button("Cancel via Graph", disabled=not (graph_ready and cancel_event_id)):
        try:
            graph_client.delete_event(cancel_event_id)
            audit_logger.log(
                action="graph_cancel_event",
                status="success",
                actor=GRAPH_SCHEDULER,
                event_id=cancel_event_id,
            )
            audit_logger.remove_event(cancel_event_id)
            st.success("Event cancelled and attendees notified.")
        except GraphClientError as e:
            st.error(f"Cancel failed: {e} ({e.details})")
            audit_logger.log(
                action="graph_cancel_failed",
                status="failed",
                actor=GRAPH_SCHEDULER,
                event_id=cancel_event_id,
                payload={"status": e.status},
                error_message=str(e),
            )

    st.markdown("---")
    st.markdown("### ICS fallback / email")

    ics_text = st.session_state.get("latest_ics")
    if ics_text:
        st.download_button(
            "Download last generated .ics",
            data=ics_text,
            file_name="interview_invite.ics",
            mime="text/calendar",
            on_click=lambda: audit_logger.log(
                action="ics_downloaded",
                status="success",
                actor=SMTP_FROM,
                event_id=st.session_state.get("latest_event_id"),
            ),
        )
        if SMTP_USER and SMTP_PASSWORD:
            to_emails = [candidate_email, hiring_manager_email]
            cc_emails = [recruiter_email_field] if recruiter_email_field else None
            if st.button("Email ICS to attendees"):
                try:
                    send_email_with_ics(to_emails, subject, invite_body.replace("<br>", "\n"), ics_text, cc_emails)
                    audit_logger.log(
                        action="smtp_sent_ics",
                        status="success",
                        actor=SMTP_FROM,
                        event_id=st.session_state.get("latest_event_id"),
                    )
                    st.success("ICS emailed to attendees.")
                except Exception as e:
                    audit_logger.log(
                        action="smtp_send_failed",
                        status="failed",
                        actor=SMTP_FROM,
                        event_id=st.session_state.get("latest_event_id"),
                        error_message=str(e),
                    )
                    st.error(f"Failed to send ICS via SMTP: {e}")
    else:
        st.info("No ICS generated yet. Submit a scheduling request or retry after a Graph failure.")

    st.markdown("---")
    st.markdown("### Audit log")
    entries = audit_logger.recent_audit_entries(limit=100)
    if entries:
        st.dataframe(entries, use_container_width=True, hide_index=True)
    else:
        st.info("No audit entries yet.")
