import ButtonToDash from "@/components/landing/ButtonToDash";
import Mockup from "@/public/mockup.png";
import Mockup2 from "@/public/mockup2.png";
import { ArchiveIcon, BoxModelIcon, CheckIcon } from "@radix-ui/react-icons";
import Image from "next/image";

export default function Home() {
  return (
    <main className="w-screen h-screen p-20 overflow-clip">
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
            The Solution to return fraud.
          </p>
        </div>

        <ButtonToDash />

        <div className="py-32">
          {bulletPoints.map((bullet, index) => (
            <div key={index} className="p-1 w-fit rounded-lg flex space-x-2">
              <CheckIcon className="w-6 h-6 bg-green-400 rounded-md text-white stroke-white stroke-[.5]" />
              <span className="font-semibold text-xl">{bullet}</span>
            </div>
          ))}
        </div>
      </div>

      <Image
        src={Mockup}
        className="invisible lg:visible absolute -top-[24dvh] -right-[60dvw] w-[1800px] aspect-auto rounded-xl -rotate-[20deg] border"
        alt="Dasbboard Mockup"
      />
      <Image
        src={Mockup2}
        className="invisible lg:visible absolute top-[8dvh] -right-[60dvw] w-[1800px] aspect-auto rounded-xl -rotate-[20deg] border"
        alt="Dasbboard Mockup"
      />
      <svg
        width="200"
        height="200"
        viewBox="0 0 200 200"
        preserveAspectRatio="xMidYMid slice"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="absolute -top-24 -left-28 -z-10 w-auto h-full artwork opacity-50"
      >
        <rect width="200" height="200" fill="rgb(255,255,255)"></rect>{" "}
        <circle
          id="blob-0"
          cx="100"
          cy="100"
          r="50"
          fill="rgb(128, 196, 255)"
          filter="url(#f0)"
        ></circle>{" "}
        <defs>
          {" "}
          <filter
            id="f0"
            x="-25%"
            y="-25%"
            width="150%"
            height="150%"
            filterUnits="userSpaceOnUse"
            color-interpolation-filters="sRGB"
          >
            <feGaussianBlur
              stdDeviation="24"
              result="fx_foregroundBlur"
            ></feGaussianBlur>{" "}
          </filter>
        </defs>{" "}
      </svg>
      <svg
        width="200"
        height="200"
        viewBox="0 0 200 200"
        preserveAspectRatio="xMidYMid slice"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="absolute -bottom-0 -right-16 -z-10 w-auto h-full artwork opacity-100"
      >
        <rect width="200" height="200" fill="rgb(255,255,255)"></rect>{" "}
        <circle
          id="blob-0"
          cx="100"
          cy="100"
          r="50"
          fill="rgb(0, 100, 0)"
          filter="url(#f0)"
        ></circle>{" "}
        <defs>
          {" "}
          <filter
            id="f0"
            x="-25%"
            y="-25%"
            width="150%"
            height="150%"
            filterUnits="userSpaceOnUse"
            color-interpolation-filters="sRGB"
          >
            <feGaussianBlur
              stdDeviation="24"
              result="fx_foregroundBlur"
            ></feGaussianBlur>{" "}
          </filter>
        </defs>{" "}
      </svg>
    </main>
  );
}

const bulletPoints = [
  "AI-Driven Item Validation & Fraud Detection",
  "Easy to Use Dashboard for Businesses",
  "Self-Training Computer Vision Algorithm",
];
