import os
import base64
import json
import re
import uuid
import imaplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from openai import OpenAI

# PDF & Image Conversion
import fitz  # PyMuPDF
from PIL import Image
import io

# =========================
#  CONFIG & CLIENT SETUP
# =========================

st.set_page_config(
    page_title="PowerDash Scheduler — Prototype",
    layout="wide",
)

# Make page a bit wider / nicer
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

# Load secrets (Streamlit Cloud) or env
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

SMTP_USER = st.secrets.get("SMTP_USER", os.environ.get("SMTP_USER", ""))
SMTP_PASSWORD = st.secrets.get("SMTP_PASSWORD", os.environ.get("SMTP_PASSWORD", ""))
SMTP_HOST = st.secrets.get("SMTP_HOST", os.environ.get("SMTP_HOST", "smtp.gmail.com"))
SMTP_PORT = int(st.secrets.get("SMTP_PORT", os.environ.get("SMTP_PORT", 587)))

IMAP_HOST = st.secrets.get("IMAP_HOST", os.environ.get("IMAP_HOST", "imap.gmail.com"))
IMAP_PORT = int(st.secrets.get("IMAP_PORT", os.environ.get("IMAP_PORT", 993)))

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY is not set in Streamlit secrets or environment.")

client = OpenAI()

# Initialise session state
if "slots" not in st.session_state:
    st.session_state["slots"] = []
if "email_body" not in st.session_state:
    st.session_state["email_body"] = ""
if "parsed_replies" not in st.session_state:
    st.session_state["parsed_replies"] = []


# =========================
#  HELPER FUNCTIONS
# =========================

def send_plain_email(to_email: str, subject: str, body: str, cc: list | None = None):
    """Send a simple plain-text email via SMTP."""
    import smtplib

    if not SMTP_USER or not SMTP_PASSWORD:
        st.error("SMTP_USER / SMTP_PASSWORD not set in secrets.")
        return

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    recipients = [to_email] + (cc or [])

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, recipients, msg.as_string())


def send_email_with_ics(
    to_emails: list[str],
    subject: str,
    body: str,
    ics_text: str,
    cc_emails: list[str] | None = None,
):
    """Send email with ICS calendar invite attached to all recipients."""
    import smtplib

    if not SMTP_USER or not SMTP_PASSWORD:
        st.error("SMTP_USER / SMTP_PASSWORD not set in secrets.")
        return

    msg = MIMEMultipart("mixed")
    msg["From"] = SMTP_USER
    msg["To"] = ", ".join(to_emails)
    if cc_emails:
        msg["Cc"] = ", ".join(cc_emails)
    msg["Subject"] = subject

    # Plain-text body
    msg.attach(MIMEText(body, "plain"))

    # ICS part
    ics_part = MIMEText(ics_text, "calendar;method=REQUEST")
    ics_part.add_header(
        "Content-Disposition", "attachment", filename="interview_invite.ics"
    )
    msg.attach(ics_part)

    recipients = to_emails + (cc_emails or [])

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, recipients, msg.as_string())


# --------------------------------------------------------
# PDF → PNG Conversion Helper (Required for GPT Vision)
# --------------------------------------------------------
def pdf_to_png(file_bytes: bytes) -> bytes:
    """Convert first page of a PDF into a PNG image (as bytes)."""
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    page = pdf.load_page(0)
    pix = page.get_pixmap(dpi=200)
    png_bytes = pix.tobytes("png")
    return png_bytes


# --------------------------------------------------------
# Calendar Parsing (PDFs + Images → OCR → Slots)
# --------------------------------------------------------
def parse_calendar(file_bytes: bytes, filename: str):
    """
    Convert PDF to PNG if needed, then call GPT-4o-mini via chat.completions
    with an image_url and text prompt. Returns a list of slot dicts.
    """

    # If PDF → convert to PNG
    if filename.lower().endswith(".pdf"):
        try:
            file_bytes = pdf_to_png(file_bytes)
            mime = "image/png"
        except Exception as e:
            st.error(f"PDF conversion failed: {e}")
            return []
    else:
        # Assume image already
        if filename.lower().endswith(".png"):
            mime = "image/png"
        else:
            mime = "image/jpeg"

    # Convert to data URL for image_url
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    prompt = """
Extract all AVAILABLE free time slots from this weekly calendar image.
Return ONLY valid JSON like:
{
  "slots": [
    {"date": "2025-11-30", "start": "09:00", "end": "09:30"},
    {"date": "2025-11-30", "start": "10:00", "end": "11:00"}
  ]
}
Only JSON. No commentary.
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You extract free time slots from calendar images and respond ONLY with valid JSON.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI for calendar parsing: {e}")
        return []

    # Decode JSON
    try:
        obj = json.loads(raw)
        if "slots" in obj and isinstance(obj["slots"], list):
            return obj["slots"]
        else:
            st.error("Model returned JSON, but missing 'slots' key.")
            st.code(raw)
            return []
    except Exception as e:
        st.error(f"Could not parse model JSON: {e}")
        st.code(raw)
        return []


def generate_scheduling_email(
    cand_name: str,
    cand_email: str,
    hm_name: str,
    company: str,
    role: str,
    cand_tz: str,
    slots: list[dict],
):
    """Use GPT to generate the scheduling email text (body only – no subject)."""
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


def check_scheduler_inbox(limit: int = 10):
    """Fetch unread messages from the scheduler mailbox via IMAP."""
    results = []
    if not SMTP_USER or not SMTP_PASSWORD:
        st.error("SMTP_USER / SMTP_PASSWORD not set – cannot check inbox.")
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
    except Exception as e:
        st.error(f"Error checking IMAP inbox: {e}")

    return results


def interpret_slot_choice(body: str, num_slots: int) -> int | None:
    """
    Very simple parser:
    - Look for integers in the email body.
    - First integer in range [1, num_slots] is treated as the chosen option.
    """
    numbers = re.findall(r"\b([1-9][0-9]?)\b", body)
    for n in numbers:
        val = int(n)
        if 1 <= val <= num_slots:
            return val
    return None


def build_ics_event(
    slot: dict,
    hm_name: str,
    hm_email: str,
    hm_tz: str,
    recruiter_name: str,
    recruiter_email: str,
    candidate_name: str,
    candidate_email: str,
    company: str,
    role: str,
    interview_type: str,
    location_or_instructions: str,
):
    """
    Build an ICS text for the chosen slot.

    - Interview type: "Teams" or "Face to face"
    - Recruiter is optional attendee.
    """

    tz = ZoneInfo(hm_tz)

    start_dt = datetime.fromisoformat(f"{slot['date']}T{slot['start']}:00").replace(
        tzinfo=tz
    )
    end_dt = datetime.fromisoformat(f"{slot['date']}T{slot['end']}:00").replace(
        tzinfo=tz
    )

    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dtstart_local = start_dt.strftime("%Y%m%dT%H%M%S")
    dtend_local = end_dt.strftime("%Y%m%dT%H%M%S")
    uid = f"{uuid.uuid4()}@powerdashhr.com"

    if interview_type == "Teams":
        summary = f"Teams Interview – {role} at {company}"
        location = "Microsoft Teams"
        desc = (
            f"Online interview via Microsoft Teams.\n\n"
            f"Joining details:\n{location_or_instructions.strip()}\n\n"
            f"Candidate: {candidate_name}\nHiring Manager: {hm_name}\nRecruiter: {recruiter_name}"
        )
    else:
        summary = f"Interview – {role} at {company}"
        location = "On-site"
        desc = (
            f"Face-to-face interview.\n\n"
            f"Location / instructions:\n{location_or_instructions.strip()}\n\n"
            f"Candidate: {candidate_name}\nHiring Manager: {hm_name}\nRecruiter: {recruiter_name}"
        )

    desc = desc.replace("\n", "\\n")

    ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//PowerDashHR//Scheduler//EN
METHOD:REQUEST
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{dtstamp}
DTSTART;TZID={hm_tz}:{dtstart_local}
DTEND;TZID={hm_tz}:{dtend_local}
SUMMARY:{summary}
LOCATION:{location}
DESCRIPTION:{desc}
ORGANIZER;CN={recruiter_name}:MAILTO:{recruiter_email}
ATTENDEE;CN={candidate_name};ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION:MAILTO:{candidate_email}
ATTENDEE;CN={hm_name};ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION:MAILTO:{hm_email}
ATTENDEE;CN={recruiter_name};ROLE=OPT-PARTICIPANT;PARTSTAT=NEEDS-ACTION:MAILTO:{recruiter_email}
END:VEVENT
END:VCALENDAR
""".strip()

    return ics


# =========================
#  UI LAYOUT
# =========================

st.title("PowerDash Scheduler — Prototype")
st.caption("Standalone scheduling assistant for in-house TA teams.")

tab_main, tab_inbox, tab_invites = st.tabs(
    ["1️⃣ New scheduling request", "2️⃣ Scheduler inbox", "3️⃣ Calendar invites"]
)

# -------------
# TAB 1: MAIN
# -------------
with tab_main:
    col_left, col_mid, col_right = st.columns([1.3, 1.0, 1.3])

    # --- Hiring manager & recruiter info ---
    with col_left:
        st.subheader("Hiring Manager & Recruiter")

        hm_name = st.text_input("Hiring manager name", value="Martin McDonald")
        hm_email = st.text_input("Hiring manager email", value="martinmcd21@hotmail.com")
        hm_tz = st.text_input(
            "Hiring manager time zone (IANA, e.g. Europe/London, America/New_York)",
            value="Europe/London",
        )

        company = st.text_input("Company name", value="PowerDash HR")
        role_title = st.text_input("Role title", value="Architect")

        st.markdown("---")
        recruiter_name = st.text_input("Recruiter name", value="Amanda Hansen")
        recruiter_email = st.text_input(
            "Recruiter email", value="info@powerdashhr.com"
        )

    # --- Upload calendar ---
    with col_mid:
        st.subheader("Upload Calendar (PDF/Image)")
        uploaded = st.file_uploader(
            "Export or print the hiring manager's free/busy view for the week (PDF, PNG, JPG).",
            type=["pdf", "png", "jpg", "jpeg"],
        )

        parse_btn = st.button("Parse availability", type="primary", disabled=not uploaded)

        if parse_btn and uploaded is not None:
            with st.spinner("Parsing calendar with GPT-4o-mini..."):
                slots = parse_calendar(uploaded.read(), uploaded.name)
                st.session_state["slots"] = slots

        slots = st.session_state.get("slots", [])

        if slots:
            st.markdown("**Detected free slots**")
            st.dataframe(slots, use_container_width=True, hide_index=True)

    # --- Candidate info & email generation ---
    with col_right:
        st.subheader("Candidate")

        cand_name = st.text_input("Candidate name", value="Ruth Nicholson")
        cand_email = st.text_input(
            "Candidate email", value="ruthnicholson1@hotmail.com"
        )
        cand_tz = st.text_input(
            "Candidate time zone (IANA, e.g. Europe/London, America/New_York)",
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
            "Email preview (from scheduler mailbox)",
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
# TAB 3: INVITES
# -------------
with tab_invites:
    st.subheader("Create & send calendar invites")

    slots = st.session_state.get("slots", [])
    parsed_replies = st.session_state.get("parsed_replies", [])

    if not slots:
        st.info("No availability slots detected yet. Parse a calendar in tab 1 first.")
    else:
        # Choose which reply to use (if any)
        reply_labels = [
            f"{i+1}. {r['subject']} — {r['from']}"
            for i, r in enumerate(parsed_replies)
        ]
        selected_reply_index = None
        if reply_labels:
            selected_label = st.selectbox(
                "Candidate reply to use (optional)",
                options=["(None – choose slot manually)"] + reply_labels,
                index=0,
            )
            if selected_label != "(None – choose slot manually)":
                selected_reply_index = reply_labels.index(selected_label)

        default_slot_index = 0
        if (
            selected_reply_index is not None
            and parsed_replies[selected_reply_index]["chosen_option"]
        ):
            opt = parsed_replies[selected_reply_index]["chosen_option"]
            if 1 <= opt <= len(slots):
                default_slot_index = opt - 1

        slot_options = [
            f"{i+1}. {s['date']} {s['start']}–{s['end']}" for i, s in enumerate(slots)
        ]
        chosen_slot_label = st.selectbox(
            "Interview slot",
            options=slot_options,
            index=default_slot_index,
        )
        chosen_slot_index = slot_options.index(chosen_slot_label)
        chosen_slot = slots[chosen_slot_index]

        st.markdown("### Interview type")

        interview_type = st.radio(
            "How will the interview take place?",
            options=["Teams", "Face to face"],
            index=0,
            horizontal=True,
        )

        if interview_type == "Teams":
            teams_link = st.text_input(
                "Teams meeting link / instructions",
                value="Paste your Teams meeting link here.",
            )
            location_text = teams_link or "Microsoft Teams (link to follow)"
        else:
            location_text = st.text_area(
                "Location & instructions (free text)",
                value="PowerDash HR Offices, 123 Example Street, London.\nPlease report to reception.",
            )

        st.markdown("### Invite details")

        invite_subject = st.text_input(
            "Calendar invite subject",
            value=f"Interview – {role_title} at {company}",
        )

        default_invite_body = (
            f"Hi all,\n\n"
            f"Interview for {role_title} at {company}.\n\n"
            f"Candidate: {cand_name}\n"
            f"Hiring manager: {hm_name}\n"
            f"Recruiter: {recruiter_name}\n\n"
            f"Best regards,\n{recruiter_name}"
        )

        invite_body = st.text_area(
            "Email body (sent with the calendar invite)",
            value=default_invite_body,
            height=220,
        )

        if st.button(
            "Generate & send calendar invites",
            type="primary",
            use_container_width=False,
            disabled=not (SMTP_USER and hm_email and cand_email and recruiter_email),
        ):
            try:
                ics_text = build_ics_event(
                    slot=chosen_slot,
                    hm_name=hm_name,
                    hm_email=hm_email,
                    hm_tz=hm_tz,
                    recruiter_name=recruiter_name,
                    recruiter_email=recruiter_email,
                    candidate_name=cand_name,
                    candidate_email=cand_email,
                    company=company,
                    role=role_title,
                    interview_type=interview_type,
                    location_or_instructions=location_text,
                )

                to_emails = [cand_email, hm_email, recruiter_email]
                send_email_with_ics(
                    to_emails=to_emails,
                    subject=invite_subject,
                    body=invite_body,
                    ics_text=ics_text,
                )

                st.success(
                    "Calendar invite sent to candidate, hiring manager, and recruiter "
                    "from the scheduling mailbox. 🎉"
                )
            except Exception as e:
                st.error(f"Error sending calendar invite: {e}")
