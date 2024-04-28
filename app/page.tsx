import ButtonToDash from "@/components/landing/ButtonToDash";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";

import { ArchiveIcon, BoxModelIcon, CheckIcon } from "@radix-ui/react-icons";
import Image from "next/image";

export default function Home() {
  return (
    <main className="w-screen h-screen p-20">
      <div className="flex items-center space-x-2 text-2xl font-bold">
        <div className="p-0.5 bg-black aspect-square w-fit rounded-lg">
          <CheckIcon className="w-7 h-7 text-white stroke-white stroke-[.5]" />
        </div>
        <span>CheckThat.</span>
      </div>

      <div className="my-10 py-10">
        <div className="text-5xl font-bold">
          <h1>
            Returns,{" "}
            <span className="bg-gradient-to-r from-blue-600 to-green-400 text-transparent bg-clip-text">
              Redefined
            </span>
            .
          </h1>
          <p className="text-xl font-[350] py-2">
            The Solution to return fraud
          </p>
        </div>
        <div className="">
          <ButtonToDash />
        </div>

        <div className="py-32">
          {bulletPoints.map((bullet) => (
            <div className="p-1 w-fit rounded-lg flex space-x-2">
              <CheckIcon className="w-6 h-6 bg-green-400 rounded-md text-white stroke-white stroke-[.5]" />
              <span className="font-semibold text-xl">{bullet}</span>
            </div>
          ))}
        </div>
      </div>
      <div className=" absolute -top-[27dvh] -right-[50dvw] aspect-video bg-black h-[100dvh] rounded-xl -rotate-[20deg] " />
    </main>
  );
}

const bulletPoints = [
  "AI-Driven Item Validation & Fraud Detection",
  "Easy to Use Dashboard for Businesses",
  "Real-Time Chatbot Assistance",
  "Self-Training Computer Vision Algorithm",
];
