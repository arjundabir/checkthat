import SignOut from "@/components/login/SignOut";
import React from "react";

const page = () => {
  return (
    <div className="container mx-auto h-screen flex flex-col items-center justify-center space-y-2">
      <h1 className="font-bold text-3xl">Thank you for Checking Us Out!</h1>
      <SignOut />
    </div>
  );
};

export default page;
