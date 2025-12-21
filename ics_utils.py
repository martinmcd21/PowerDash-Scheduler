import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional


def format_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_ics_invite(
    start_dt: datetime,
    end_dt: datetime,
    subject: str,
    description: str,
    organizer_email: str,
    organizer_name: str,
    attendees: List[tuple[str, str]],
    location: str,
    uid: Optional[str] = None,
) -> str:
    uid_value = uid or f"{uuid.uuid4()}@powerdashhr.com"
    dtstamp = format_dt(datetime.utcnow())
    dtstart = format_dt(start_dt)
    dtend = format_dt(end_dt)

    attendee_lines = []
    for name, email in attendees:
        attendee_lines.append(
            f"ATTENDEE;CN={name};ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION:MAILTO:{email}"
        )

    attendees_block = "\n".join(attendee_lines)

    ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//PowerDashHR//Scheduler//EN
METHOD:REQUEST
BEGIN:VEVENT
UID:{uid_value}
DTSTAMP:{dtstamp}
DTSTART:{dtstart}
DTEND:{dtend}
SUMMARY:{subject}
DESCRIPTION:{description}
ORGANIZER;CN={organizer_name}:MAILTO:{organizer_email}
{attendees_block}
LOCATION:{location}
END:VEVENT
END:VCALENDAR
""".strip()
    return ics


def build_ics_cancellation(
    start_dt: datetime,
    subject: str,
    organizer_email: str,
    organizer_name: str,
    attendees: List[tuple[str, str]],
    uid: str,
) -> str:
    dtstamp = format_dt(datetime.utcnow())
    attendee_lines = []
    for name, email in attendees:
        attendee_lines.append(
            f"ATTENDEE;CN={name};ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION:MAILTO:{email}"
        )
    attendees_block = "\n".join(attendee_lines)
    ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//PowerDashHR//Scheduler//EN
METHOD:CANCEL
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{dtstamp}
SUMMARY:{subject}
ORGANIZER;CN={organizer_name}:MAILTO:{organizer_email}
{attendees_block}
STATUS:CANCELLED
DTSTART:{format_dt(start_dt)}
END:VEVENT
END:VCALENDAR
""".strip()
    return ics


def build_sample_invite() -> str:
    now = datetime.now(timezone.utc)
    return build_ics_invite(
        start_dt=now + timedelta(days=1),
        end_dt=now + timedelta(days=1, hours=1),
        subject="Sample Interview",
        description="Sample event",
        organizer_email="scheduling@powerdashhr.com",
        organizer_name="Scheduler Bot",
        attendees=[("Candidate", "candidate@example.com")],
        location="Microsoft Teams",
    )
