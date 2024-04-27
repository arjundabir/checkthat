import React from "react";
import { Input } from "@/components/ui/input";
import { MagnifyingGlassIcon } from "@radix-ui/react-icons";

const Navbar = () => {
  return (
    <div className="w-screen h-20 border ">
      <div className="w-full h-full container mx-auto flex justify-between items-center">
        <h2 className="font-bold">CheckThat!</h2>
        <div className="w-96 flex items-center border rounded-lg">
          <MagnifyingGlassIcon className="" width={35} />
          <Input
            className="w-full py-4 border-0 outline-none ring-0 shadow-none focus-visible:ring-0"
            placeholder="Search for products"
          />
        </div>
        <div className="h-1/2 aspect-square p-2 border rounded-lg flex justify-center items-center">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 576 512"
            className="w-4"
          >
            <path d="M0 24C0 10.7 10.7 0 24 0H69.5c22 0 41.5 12.8 50.6 32h411c26.3 0 45.5 25 38.6 50.4l-41 152.3c-8.5 31.4-37 53.3-69.5 53.3H170.7l5.4 28.5c2.2 11.3 12.1 19.5 23.6 19.5H488c13.3 0 24 10.7 24 24s-10.7 24-24 24H199.7c-34.6 0-64.3-24.6-70.7-58.5L77.4 54.5c-.7-3.8-4-6.5-7.9-6.5H24C10.7 48 0 37.3 0 24zM128 464a48 48 0 1 1 96 0 48 48 0 1 1 -96 0zm336-48a48 48 0 1 1 0 96 48 48 0 1 1 0-96z" />
          </svg>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
