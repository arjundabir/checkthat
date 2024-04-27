import React from "react";
import { Input } from "@/components/ui/input";
import { MagnifyingGlassIcon } from "@radix-ui/react-icons";

const Navbar = () => {
  return (
    <div className="w-screen h-20 border flex justify-center items-center">
      <div className="w-96 flex items-center border rounded-lg">
        <MagnifyingGlassIcon className="" width={35} />
        <Input
          className="w-full py-4 border-0 outline-none ring-0 shadow-none focus-visible:ring-0"
          placeholder="Search for products"
        />
      </div>
    </div>
  );
};

export default Navbar;
