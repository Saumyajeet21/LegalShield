import httpx
from fastapi import HTTPException

from .config import settings

BUCKET = "legalshield-originals"
PUBLIC_DOCUMENT = "id,name,media_type,category,sha256,page_count,finding_count,status,synthetic,trusted,revision,created_at"


class Store:
    def __init__(self, client: httpx.AsyncClient, token: str):
        self.client = client
        self.token = token
        self.config = settings()
        self.anon = self.config.supabase_anon_key.get_secret_value()
        self.service = self.config.supabase_service_role_key.get_secret_value()
        self.owner = ""
        self.researcher = False

    async def http(self, method, path, *, privileged=False, binary=False, allow_missing=False, headers=None, **kwargs):
        if not self.config.supabase_url or not self.anon or not self.service:
            raise HTTPException(503, "Configure the Supabase URL, publishable key and server-only service role key.")
        key, bearer = (self.service, self.service) if privileged else (self.anon, self.token)
        # Secret API keys are authenticated by the apikey header at the Supabase gateway.
        security = {"apikey": key}
        if not privileged or not key.startswith("sb_secret_"):
            security["Authorization"] = f"Bearer {bearer}"
        try:
            result = await self.client.request(method, f"{self.config.supabase_url.rstrip('/')}/{path}",
                                               headers={**security, **(headers or {})}, **kwargs)
        except httpx.RequestError:
            raise HTTPException(503, "Supabase is unreachable. Check the project URL and connectivity.")
        if allow_missing and result.status_code in (400, 404):
            try:
                code = result.json().get("statusCode", result.status_code)
            except ValueError:
                code = result.status_code
            if str(code) == "404":
                return None
        if result.status_code in (401, 403) and path == "auth/v1/user":
            raise HTTPException(401, "Sign in again to use your legal workspace.")
        if not result.is_success:
            raise HTTPException(502, "Supabase rejected the operation. Check the migration, storage bucket, and keys.")
        if binary:
            return result.content
        return result.json() if result.content else None

    async def authenticate(self):
        user = await self.http("GET", "auth/v1/user")
        self.owner = user["id"]
        self.researcher = user.get("app_metadata", {}).get("legalshield_role") == "researcher"
        return self

    async def read(self, table, **params):
        return await self.http("GET", f"rest/v1/{table}", params={"owner": f"eq.{self.owner}", **params})

    async def write(self, method, table, *, data=None, **params):
        if method == "POST":
            data = {**data, "owner": self.owner}
        return await self.http(method, f"rest/v1/{table}", privileged=True,
                               params={"owner": f"eq.{self.owner}", **params} if method != "POST" else params,
                               headers={"Prefer": "return=representation"}, json=data)

    async def rpc(self, name, data, privileged=False):
        return await self.http("POST", f"rest/v1/rpc/legalshield_{name}", privileged=privileged, json=data)

    async def put_original(self, path, data, kind):
        await self.http("POST", f"storage/v1/object/{BUCKET}/{path}", privileged=True, content=data,
                        headers={"Content-Type": kind, "x-upsert": "false"})

    async def get_original(self, path):
        return await self.http("GET", f"storage/v1/object/{BUCKET}/{path}", privileged=True, binary=True)

    async def remove_original(self, path):
        # Deleting a missing prefix is idempotent in the batch deletion API.
        return await self.http("DELETE", f"storage/v1/object/{BUCKET}", privileged=True, json={"prefixes": [path]})
