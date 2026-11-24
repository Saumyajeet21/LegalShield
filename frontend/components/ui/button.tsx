import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import type { ComponentProps } from "react";

const buttonVariants = cva("button", {
  variants: { variant: { default: "button-primary", outline: "button-outline", ghost: "button-ghost", danger: "button-danger" } },
  defaultVariants: { variant: "default" }
});

export function Button({ className, variant, type = "button", ...props }: ComponentProps<"button"> & VariantProps<typeof buttonVariants>) {
  return <button type={type} className={cn(buttonVariants({ variant }), className)} {...props} />;
}
