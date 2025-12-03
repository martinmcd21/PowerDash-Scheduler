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

# Image + PDF
from pdf2image import convert_from_bytes
from PIL import Image
import io


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="PowerDash Scheduler — Prototype",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1450px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD SECRETS
# ============================================================

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

SMTP_USER = st.secrets.get("SMTP_USER", os.environ.get("SMTP_USER", ""))
SMTP_PASSWORD = st.secrets.get("SMTP_PASSWORD", os.environ.get("SMTP_PASSWORD", ""))
SMTP_HOST = st.secrets.get("SMTP_HOST", os.environ.get("SMTP_HOST", "smtp.gmail.com"))
SMTP_PORT = int(st.secrets.get("SMTP_PORT", os.environ.get("SMTP_PORT", 587)))

IMAP_HOST = st.secrets.get("IMAP_HOST", os.environ.get("IMAP_HOST", "imap.gmail.com"))
IMAP_PORT = int(st.secrets.get("IMAP_PORT", os.environ.get("IMAP_PORT", 993)))

client = OpenAI()

# State
if "slots" not in st.session_state:
    st.session_state["slots"] = []

if "email_body" not in st.session_state:
    st.session_state["email_body"] = ""

if "parsed_replies" not in st.session_state:
    st.session_state["parsed_replies"] = []


# ============================================================
# EMAIL SENDING
# ============================================================

def send_plain_email(to_email, subject, body, cc=None):
    import smtplib

    if not SMTP_USER or not SMTP_PASSWORD:
        st.error("SMTP credentials missing.")
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


def send_email_with_ics(to_emails, subject, body, ics_text, cc_emails=None):
    import smtplib

    msg = MIMEMultipart("mixed")
    msg["From"] = SMTP_USER
    msg["To"] = ", ".join(to_emails)
    if cc_emails:
        msg["Cc"] = ", ".join(cc_emails)
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    ics_part = MIMEText(ics_text, "calendar;method=REQUEST")
    ics_part.add_header("Content-Disposition", "attachment", filename="invite.ics")
    msg.attach(ics_part)

    recipients = to_emails + (cc_emails or [])

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, recipients, msg.as_string())


# ============================================================
# PDF → PNG CONVERSION
# ============================================================

def pdf_to_png(file_bytes):
    pages = convert_from_bytes(file_bytes, dpi=200)
    img = pages[0]
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


# ============================================================
# CALENDAR PARSER (PDF/PNG/JPG → GPT)
# ============================================================

def parse_calendar(file_bytes, filename):
    # PDF → PNG
    if filename.lower().endswith(".pdf"):
        try:
            file_bytes = pdf_to_png(file_bytes)
            mime = "image/png"
        except Exception as e:
            st.error(f"PDF conversion failed: {e}")
            return []
    else:
        ext = filename.lower().split(".")[-1]
        mime = "image/png" if ext == "png" else "image/jpeg"

    b64 = base64.b64encode(file_bytes).decode("utf-8")

    prompt = """
Extract AVAILABLE free/busy slots from this weekly calendar image.

Return ONLY valid JSON of the shape:
{
  "slots": [
    {"date": "2025-11-30", "start": "09:00", "end": "09:30"},
    {"date": "2025-11-30", "start": "10:00", "end": "11:00"}
  ]
}
No commentary.
"""

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image": {"data": b64, "mime_type": mime}}
            ],
            max_output_tokens=500,
        )
        raw = resp.output_text
    except Exception as e:
        st.error(f"OpenAI parse error: {e}")
        return []

    try:
        data = json.loads(raw)
        return data.get("slots", [])
    except Exception:
        st.error("Model returned invalid JSON.")
        st.code(raw)
        return []


# ============================================================
# EMAIL GENERATION (GPT)
# ============================================================

def generate_scheduling_email(cand_name, cand_email, hm_name, company, role, cand_tz, slots):
    if not slots:
        return "No slots available."

    slot_lines = [
        f"{i+1}. {s['date']} {s['start']}–{s['end']} ({cand_tz})"
        for i, s in enumerate(slots)
    ]
    slot_text = "\n".join(slot_lines)

    prompt = f"""
Write a warm, concise scheduling email to the candidate.

Candidate: {cand_name} <{cand_email}>
Hiring manager: {hm_name}
Company: {company}
Role: {role}
Candidate timezone: {cand_tz}

Interview options:
{slot_text}

Instructions:
- Label each option.
- Ask them to reply ONLY with the option number.
- Professional & friendly.
- No subject line.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You write friendly professional recruiting emails."},
            {"role": "user", "content": prompt},
        ],
    )

    return resp.choices[0].message.content.strip()


# ============================================================
# INBOX PARSING
# ============================================================

def check_scheduler_inbox(limit=10):
    msgs = []

    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        mail.login(SMTP_USER, SMTP_PASSWORD)
        mail.select("INBOX")

        _, data = mail.search(None, "UNSEEN")
        ids = data[0].split()

        for msg_id in ids[-limit:]:
            _, msg_data = mail.fetch(msg_id, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])

            from_addr = email.utils.parseaddr(msg.get("From", ""))[1]
            subject = msg.get("Subject", "")
            body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = (part.get_payload(decode=True) or b"").decode(errors="ignore")
                        break
            else:
                body = (msg.get_payload(decode=True) or b"").decode(errors="ignore")

            msgs.append({"from": from_addr, "subject": subject, "body": body})

        mail.logout()
    except Exception as e:
        st.error(f"IMAP error: {e}")

    return msgs


def interpret_slot_choice(body, num_slots):
    numbers = re.findall(r"\b([0-9]+)\b", body)
    for n in numbers:
        v = int(n)
        if 1 <= v <= num_slots:
            return v
    return None


# ============================================================
# ICS INVITE BUILDER
# ============================================================

def build_ics_event(slot, hm_name, hm_email, hm_tz,
                    recruiter_name, recruiter_email,
                    cand_name, cand_email,
                    company, role, interview_type,
                    location_or_instructions):

    tz = ZoneInfo(hm_tz)

    start_dt = datetime.fromisoformat(f"{slot['date']}T{slot['start']}:00").replace(tzinfo=tz)
    end_dt = datetime.fromisoformat(f"{slot['date']}T{slot['end']}:00").replace(tzinfo=tz)

    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dtstart_local = start_dt.strftime("%Y%m%dT%H%M%S")
    dtend_local = end_dt.strftime("%Y%m%dT%H%M%S")
    uid = f"{uuid.uuid4()}@powerdashhr.com"

    if interview_type == "Teams":
        summary = f"Teams Interview – {role} at {company}"
        location = "Microsoft Teams"
        desc = f"Online interview.\n\nTeams link:\n{location_or_instructions}"
    else:
        summary = f"Interview – {role} at {company}"
        location = "On-site"
        desc = f"Face-to-face interview.\n\nLocation:\n{location_or_instructions}"

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
ATTENDEE;CN={cand_name};ROLE=REQ-PARTICIPANT:MAILTO:{cand_email}
ATTENDEE;CN={hm_name};ROLE=REQ-PARTICIPANT:MAILTO:{hm_email}
ATTENDEE;CN={recruiter_name};ROLE=OPT-PARTICIPANT:MAILTO:{recruiter_email}
END:VEVENT
END:VCALENDAR"""
    return ics


# ============================================================
# UI TABS
# ============================================================

st.title("PowerDash Scheduler — Prototype")
st.caption("Standalone scheduling assistant for in-house TA teams.")

tab_main, tab_inbox, tab_invites = st.tabs(
    ["1️⃣ New scheduling request", "2️⃣ Scheduler inbox", "3️⃣ Calendar invites"]
)


# ------------------------------------------------------------
# TAB 1 — Scheduling Setup
# ------------------------------------------------------------
with tab_main:
    col_left, col_mid, col_right = st.columns([1.3, 1.0, 1.3])

    # LEFT — Hiring manager & recruiter
    with col_left:
        st.subheader("Hiring Manager & Recruiter")

        hm_name = st.text_input("Hiring manager name", value="Martin McDonald")
        hm_email = st.text_input("Hiring manager email", value="martinmcd21@hotmail.com")
        hm_tz = st.text_input("Hiring manager timezone (IANA)", value="Europe/London")

        company = st.text_input("Company", value="PowerDash HR")
        role_title = st.text_input("Role", value="Architect")

        st.markdown("---")
        recruiter_name = st.text_input("Recruiter name", value="Amanda Hansen")
        recruiter_email = st.text_input("Recruiter email", value="info@powerdashhr.com")

    # MIDDLE — Upload calendar
    with col_mid:
        st.subheader("Upload Calendar (PDF/Image)")

        uploaded = st.file_uploader(
            "Upload PDF, PNG, JPG of hiring manager's free/busy.",
            type=["pdf", "png", "jpg", "jpeg"],
        )

        parse_btn = st.button("Parse availability", disabled=not uploaded)

        if parse_btn:
            with st.spinner("Parsing calendar..."):
                slots = parse_calendar(uploaded.read(), uploaded.name)
                st.session_state["slots"] = slots

        if st.session_state["slots"]:
            st.markdown("**Detected free slots**")
            st.dataframe(st.session_state["slots"], use_container_width=True)

    # RIGHT — Candidate email
    with col_right:
        st.subheader("Candidate")

        cand_name = st.text_input("Candidate name", value="Ruth Nicholson")
        cand_email = st.text_input("Candidate email", value="candidate@example.com")
        cand_tz = st.text_input("Candidate timezone (IANA)", value="Europe/London")

        st.markdown("### Scheduling email")

        gen_email_btn = st.button("Generate scheduling email", disabled=not st.session_state["slots"])
        if gen_email_btn:
            with st.spinner("Generating email..."):
                st.session_state["email_body"] = generate_scheduling_email(
                    cand_name, cand_email, hm_name,
                    company, role_title, cand_tz,
                    st.session_state["slots"],
                )

        email_body = st.text_area(
            "Email preview",
            value=st.session_state["email_body"],
            height=250,
        )

        if st.button("Send email to candidate", disabled=not email_body):
            try:
                send_plain_email(
                    cand_email,
                    subject=f"Interview availability – {role_title} at {company}",
                    body=email_body,
                    cc=[recruiter_email],
                )
                st.success("Email sent!")
            except Exception as e:
                st.error(f"Send error: {e}")


# ------------------------------------------------------------
# TAB 2 — Inbox
# ------------------------------------------------------------
with tab_inbox:
    st.subheader("Scheduler Inbox (unread replies)")

    if st.button("Check scheduler inbox"):
        with st.spinner("Checking IMAP inbox..."):
            msgs = check_scheduler_inbox(limit=10)

        parsed = []
        slots = st.session_state["slots"]
        for m in msgs:
            opt = interpret_slot_choice(m["body"], len(slots))
            parsed.append({
                "from": m["from"],
                "subject": m["subject"],
                "body": m["body"],
                "chosen_option": opt,
            })

        st.session_state["parsed_replies"] = parsed

    replies = st.session_state["parsed_replies"]

    if not replies:
        st.info("No replies yet.")
    else:
        st.success(f"Fetched {len(replies)} messages")
        for i, r in enumerate(replies):
            with st.expander(
                f"{i+1}. {r['subject']} — {r['from']} "
                f"({r['chosen_option'] or 'no option detected'})"
            ):
                st.text(r["body"])


# ------------------------------------------------------------
# TAB 3 — Calendar Invites
# ------------------------------------------------------------
with tab_invites:
    st.subheader("Create & send calendar invites")

    slots = st.session_state["slots"]
    replies = st.session_state["parsed_replies"]

    if not slots:
        st.info("Parse a calendar first.")
        st.stop()

    # Choose slot based on reply
    reply_labels = [
        f"{i+1}. {r['subject']} — {r['from']}"
        for i, r in enumerate(replies)
    ]
    selected_reply = st.selectbox(
        "Candidate reply (optional)",
        ["(None – choose manually)"] + reply_labels,
        index=0,
    )

    default_index = 0
    if selected_reply != "(None – choose manually)":
        idx = reply_labels.index(selected_reply)
        chosen = replies[idx]["chosen_option"]
        if chosen:
            default_index = chosen - 1

    slot_labels = [
        f"{i+1}. {s['date']} {s['start']}–{s['end']}"
        for i, s in enumerate(slots)
    ]

    chosen_label = st.selectbox("Interview slot", slot_labels, index=default_index)
    chosen_slot = slots[slot_labels.index(chosen_label)]

    st.markdown("### Interview type")
    interview_type = st.radio("Choose:", ["Teams", "Face to face"], horizontal=True)

    if interview_type == "Teams":
        location_text = st.text_input("Teams link", value="Paste Teams meeting link here.")
    else:
        location_text = st.text_area(
            "Location & instructions",
            value="PowerDash HR Offices, 123 Example Street, London.",
        )

    st.markdown("### Invite details")
    invite_subject = st.text_input(
        "Invite subject",
        value=f"Interview – {role_title} at {company}",
    )

    default_body = (
        f"Hi all,\n\nInterview for {role_title} at {company}.\n\n"
        f"Candidate: {cand_name}\n"
        f"Hiring Manager: {hm_name}\n"
        f"Recruiter: {recruiter_name}\n\n"
        "Best regards,\n"
        f"{recruiter_name}"
    )

    invite_body = st.text_area("Email body", value=default_body, height=200)

    if st.button("Generate & send invites", type="primary"):
        try:
            ics = build_ics_event(
                chosen_slot, hm_name, hm_email, hm_tz,
                recruiter_name, recruiter_email,
                cand_name, cand_email,
                company, role_title,
                interview_type, location_text,
            )

            to_emails = [cand_email, hm_email, recruiter_email]

            send_email_with_ics(
                to_emails=to_emails,
                subject=invite_subject,
                body=invite_body,
                ics_text=ics,
            )

            st.success("Calendar invites sent!")
        except Exception as e:
            st.error(f"Invite send error: {e}")
