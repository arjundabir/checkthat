"use client";
import React from "react";
import { useSession } from "next-auth/react";
const UserTitle = () => {
  const { status, data: session } = useSession();

  if (status === "loading") return null;
  return <div>{status === "authenticated" && session.user?.name}</div>;
};

export default UserTitle;
