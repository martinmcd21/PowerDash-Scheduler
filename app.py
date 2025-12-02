import streamlit as st
import base64
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import smtplib
import ssl
import json
from datetime import datetime, timedelta
from ics import Calendar, Event
from openai import OpenAI
from bs4 import BeautifulSoup

# Load secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SMTP_USER = st.secrets["SCHEDULER_SMTP_USER"]
SMTP_PASSWORD = st.secrets["SCHEDULER_SMTP_PASSWORD"]

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------
# EMAIL SENDER
# ---------------------------------------------------------
def send_email(to_email, subject, body, attachment_bytes=None, attachment_name=None):
    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    if attachment_bytes:
        part = MIMEApplication(attachment_bytes, Name=attachment_name)
        part['Content-Disposition'] = f'attachment; filename="{attachment_name}"'
        msg.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)


# ---------------------------------------------------------
# PARSE CALENDAR (PDF/PNG/JPG) WITH GPT-4o-mini
# ---------------------------------------------------------
def parse_calendar(file_bytes, filename):
    b64 = base64.b64encode(file_bytes).decode("utf-8")

    prompt = """
Extract all available free/busy slots from this calendar image. 
Return STRICT JSON:

{
  "slots": [
    {"date": "YYYY-MM-DD", "start": "HH:MM", "end": "HH:MM"},
    ...
  ]
}

Only JSON. No commentary.
"""

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "user", "content": prompt},
            {"role": "user", "input_image": {"data": b64}}
        ]
    )

    text = resp.output_text.strip()

    try:
        parsed = json.loads(text)
        return parsed.get("slots", [])
    except:
        st.error("Model did not return valid JSON.")
        return []


# ---------------------------------------------------------
# GENERATE SCHEDULING EMAIL
# ---------------------------------------------------------
def generate_email(cand_name, hm_name, company, role, cand_tz, slots):

    slot_text = "\n".join(
        [f"{i+1}. {s['date']} • {s['start']}-{s['end']} ({cand_tz})"
         for i, s in enumerate(slots)]
    )

    prompt = f"""
Write a friendly professional interview scheduling email.

Candidate: {cand_name}
Hiring Manager: {hm_name}
Company: {company}
Role: {role}

Use these options:

{slot_text}

Do NOT include a subject line.
"""

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return resp.output_text.strip()


# ---------------------------------------------------------
# BUILD ICS INVITE
# ---------------------------------------------------------
def build_ics(slot, hm_name, cand_name, company, mode, teams_link, f2f_location):

    event = Event()
    event.name = f"Interview – {company}"
    start_dt = datetime.fromisoformat(slot["date"] + "T" + slot["start"] + ":00")
    end_dt = datetime.fromisoformat(slot["date"] + "T" + slot["end"] + ":00")

    event.begin = start_dt
    event.end = end_dt

    if mode == "Teams":
        event.description = f"Microsoft Teams interview.\n\nJoin link:\n{teams_link}"
        event.location = "Microsoft Teams"
    else:
        event.description = f"Face-to-face interview.\n\nLocation:\n{f2f_location}"
        event.location = f2f_location

    cal = Calendar()
    cal.events.add(event)
    return str(cal).encode("utf-8")


# ---------------------------------------------------------
# READ SCHEDULER INBOX (IMAP)
# ---------------------------------------------------------
def read_inbox():
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(SMTP_USER, SMTP_PASSWORD)
    mail.select("inbox")

    typ, data = mail.search(None, "UNSEEN")
    messages = data[0].split()

    results = []

    for num in messages:
        typ, msg_data = mail.fetch(num, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        from_ = msg["From"]
        subject = msg["Subject"]
        date = msg["Date"]

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors="ignore")
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")

        results.append({
            "from": from_,
            "subject": subject,
            "date": date,
            "body": body
        })

    mail.logout()
    return results


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("PowerDash Scheduler – Prototype")

# LEFT COLUMN
left, mid, right = st.columns([1,1,1])

with left:
    st.header("Hiring Manager & Recruiter")
    hm_name = st.text_input("Hiring manager name")
    hm_email = st.text_input("Hiring manager email")
    hm_tz = st.text_input("Hiring manager time zone (IANA)")
    company = st.text_input("Company name")
    rec_name = st.text_input("Recruiter name")
    rec_email = st.text_input("Recruiter email")

with mid:
    st.header("Upload Calendar (PDF/Image)")
    uploaded = st.file_uploader("Upload", type=["pdf","png","jpg","jpeg"])

    if uploaded:
        if st.button("Parse availability"):
            slots = parse_calendar(uploaded.read(), uploaded.name)
            st.session_state["slots"] = slots

    if "slots" in st.session_state:
        st.subheader("Detected free slots")
        st.dataframe(st.session_state["slots"])

with right:
    st.header("Candidate")
    cand_name = st.text_input("Candidate name")
    cand_email = st.text_input("Candidate email")
    cand_tz = st.text_input("Candidate time zone")
    role = st.text_input("Role title")

    if "slots" in st.session_state:
        if st.button("Generate scheduling email"):
            email_body = generate_email(cand_name, hm_name, company, role, cand_tz, st.session_state["slots"])
            st.session_state["email_body"] = email_body

    if "email_body" in st.session_state:
        st.subheader("Email preview")
        edited = st.text_area("Edit before sending", st.session_state["email_body"])

        if st.button("Send email to candidate"):
            send_email(cand_email, f"Interview availability for {role}", edited)
            st.success("Sent!")

# ---------------------------------------------------------
# INBOX READER
# ---------------------------------------------------------
st.markdown("---")
st.header("Scheduler Inbox (unread replies)")

if st.button("Check scheduler inbox now"):
    msgs = read_inbox()
    st.success(f"Fetched {len(msgs)} unread messages.")
    for m in msgs:
        with st.expander(f"{m['subject']} – {m['from']}"):
            st.write("Date:", m["date"])
            st.code(m["body"])
