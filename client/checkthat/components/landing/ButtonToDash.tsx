"use client";
import React from "react";
import { Button } from "../ui/button";
import { ArchiveIcon } from "@radix-ui/react-icons";
import { useRouter } from "next/navigation";
import { signIn } from "next-auth/react";

const ButtonToDash = () => {
  const router = useRouter();

  const handleClick = () => {
    signIn("github", { callbackUrl: "/admin/dashboard" });
  };
  return (
    <div>
      <Button className="my-2" onClick={handleClick}>
        <ArchiveIcon className="mr-2 h-4 w-4" /> Access Dashboard
      </Button>
    </div>
  );
};

export default ButtonToDash;
