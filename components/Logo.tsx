import { CheckIcon } from "@radix-ui/react-icons";
import React from "react";

const Logo = () => {
  return (
    <div className="p-1 w-fit rounded-lg flex space-x-2">
      <CheckIcon className="w-6 h-6 bg-black rounded-md text-white stroke-white stroke-[.5]" />
      <span className="font-semibold text-xl">CheckThat.</span>
    </div>
  );
};

export default Logo;
