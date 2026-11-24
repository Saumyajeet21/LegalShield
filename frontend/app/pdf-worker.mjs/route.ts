import { readFile } from "node:fs/promises";
import path from "node:path";
export const dynamic = "force-static";
export async function GET() {
  const data = await readFile(path.join(process.cwd(), "node_modules/pdfjs-dist/build/pdf.worker.min.mjs"));
  return new Response(new Uint8Array(data), { headers: { "Content-Type": "text/javascript; charset=utf-8" } });
}
