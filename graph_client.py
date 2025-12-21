import time
from typing import Any, Dict, Optional

import requests


class GraphClientError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, details: Any | None = None):
        super().__init__(message)
        self.status = status
        self.details = details


class GraphClient:
    def __init__(self, tenant_id: str, client_id: str, client_secret: str, scheduler_mailbox: str):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.scheduler_mailbox = scheduler_mailbox
        self._token: dict[str, Any] | None = None

    @property
    def configured(self) -> bool:
        return bool(self.tenant_id and self.client_id and self.client_secret and self.scheduler_mailbox)

    def _token_expired(self) -> bool:
        if not self._token:
            return True
        return time.time() >= self._token.get("expires_at", 0)

    def get_access_token(self) -> str:
        if not self.configured:
            raise GraphClientError("Graph credentials are not fully configured.")

        if self._token and not self._token_expired():
            return self._token["access_token"]

        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "https://graph.microsoft.com/.default",
        }
        response = requests.post(token_url, data=data, timeout=30)
        if response.status_code != 200:
            raise GraphClientError(
                "Failed to acquire Graph token",
                status=response.status_code,
                details=response.text,
            )
        token_data = response.json()
        expires_in = token_data.get("expires_in", 0)
        token_data["expires_at"] = time.time() + expires_in - 60
        self._token = token_data
        return token_data["access_token"]

    def _headers(self) -> Dict[str, str]:
        token = self.get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": 'outlook.timezone="UTC"',
        }

    def create_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"https://graph.microsoft.com/v1.0/users/{self.scheduler_mailbox}/events?sendUpdates=all"
        response = requests.post(url, headers=self._headers(), json=payload, timeout=30)
        if response.status_code not in (200, 201):
            raise GraphClientError(
                "Graph event creation failed",
                status=response.status_code,
                details=response.text,
            )
        return response.json()

    def update_event(self, event_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"https://graph.microsoft.com/v1.0/users/{self.scheduler_mailbox}/events/{event_id}?sendUpdates=all"
        response = requests.patch(url, headers=self._headers(), json=payload, timeout=30)
        if response.status_code not in (200, 202):
            raise GraphClientError(
                "Graph event update failed",
                status=response.status_code,
                details=response.text,
            )
        return response.json() if response.text else {}

    def delete_event(self, event_id: str) -> None:
        url = f"https://graph.microsoft.com/v1.0/users/{self.scheduler_mailbox}/events/{event_id}?sendUpdates=all"
        response = requests.delete(url, headers=self._headers(), timeout=30)
        if response.status_code not in (204, 202):
            raise GraphClientError(
                "Graph event cancellation failed",
                status=response.status_code,
                details=response.text,
            )

    def list_upcoming_events(self, top: int = 5) -> Dict[str, Any]:
        url = (
            f"https://graph.microsoft.com/v1.0/users/{self.scheduler_mailbox}/calendar/events"
            f"?$orderby=start/dateTime&$top={top}"
        )
        response = requests.get(url, headers=self._headers(), timeout=30)
        if response.status_code != 200:
            raise GraphClientError(
                "Graph calendar fetch failed",
                status=response.status_code,
                details=response.text,
            )
        return response.json()

    def diagnostics(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        token_ok = False
        try:
            token = self.get_access_token()
            token_ok = True
            results["token"] = "ok" if token else "missing"
        except GraphClientError as e:
            results["token"] = {"error": str(e), "details": e.details}

        if token_ok:
            try:
                events = self.list_upcoming_events(top=3)
                results["calendar"] = {
                    "count": len(events.get("value", [])),
                    "preview": events.get("value", []),
                }
            except GraphClientError as e:
                results["calendar"] = {"error": str(e), "details": e.details}

        return results
