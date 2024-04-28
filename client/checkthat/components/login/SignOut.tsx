"use client";
import React from "react";
import { signOut } from "next-auth/react";
import { Button } from "@/components/ui/button";
import Link from "next/link";

const SignOut = () => {
  return (
    <div className="w-full flex justify-center gap-x-2">
      <Link href="/admin/dashboard">
        <Button variant="default">Back to Dashboard</Button>
      </Link>
      <Button onClick={() => signOut({ callbackUrl: "/" })} variant="secondary">
        Sign out
      </Button>
    </div>
  );
};

export default SignOut;
