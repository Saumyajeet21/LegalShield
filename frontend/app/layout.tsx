import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LegalShield AI",
  description: "Document-grounded legal research and security review.",
  icons: {icon: "/icon.svg"}
};

export default function RootLayout({ children }: Readonly<{children: React.ReactNode}>) {
  return <html lang="en"><body>{children}</body></html>;
}
