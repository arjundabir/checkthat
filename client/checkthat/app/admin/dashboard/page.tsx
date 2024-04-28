import Link from "next/link";
import React from "react";
import {
  Logo,
  SettingsIcon,
  UsersIcon,
  VercelLogo,
} from "@/components/dashboard/icons";
import { NavItem } from "@/components/dashboard/nav-item";
import { DataTableDemo } from "@/components/dashboard/Table";

const page = () => {
  return (
    <main className="px-5 py-6">
      <DataTableDemo />
    </main>
  );
};

export default page;
