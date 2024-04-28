"use client";
import React, { useEffect, useState } from "react";
import { Button } from "../ui/button";
import { Progress } from "../ui/progress";
import Logo from "../Logo";
import { BarChartIcon, CheckIcon, Cross2Icon } from "@radix-ui/react-icons";
import { DataTableDemo } from "./Table";

interface Data {
  _id: string;
  image: string;
  condition: string;
  authenticity: string;
  packaging_and_accessories: string;
  damage: string;
  returnable: string;
}

interface Props {
  uuid: string;
  IndividualReturnBox: Boolean;
  setIRB: (value: React.SetStateAction<Boolean>) => void;
}

const IndividualItem = ({ uuid, IndividualReturnBox, setIRB }: Props) => {
  const [response, setResponse] = useState<Data>();
  const [randomVar, setRandomVar] = useState<number>();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("/api/fetch");
        if (!response.ok) {
          // Check if response status is not okay, which indicates a server error.
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: Data = await response.json();
        setResponse(data);
        if (!data) {
          // If no data, log and set to retry
          console.log("No data received, retrying in 5 seconds...");
          setTimeout(fetchData, 5000); // Wait for 5 seconds and retry
        } else {
          console.log(data);
          // Continue with your logic here
        }
      } catch (error) {
        console.error("Error:", error);
        console.log("Attempt failed, retrying in 5 seconds...");
        setTimeout(fetchData, 5000); // Wait for 5 seconds and retry
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    const value = parseInt(response?.returnable || "");
    value == 0
      ? setRandomVar(Math.random() * 0.5)
      : setRandomVar(0.5 + Math.random() * 0.5);
  }, [response]);
  return (
    <>
      {!response ? (
        <div>Loading</div>
      ) : (
        <>
          <h2 className="text-2xl">
            Order: <span className="font-semibold">{uuid}</span>
          </h2>
          <div className="w-full rounded-lg my-2">
            <div className="">
              <div className="flex gap-4 h-auto">
                <div className="flex flex-col gap-2">
                  <div className="w-full h-full self-start border rounded-lg p-4">
                    <img
                      src={response?.image}
                      alt="Scanned Image"
                      width={1366}
                      height={768}
                      className="flex-1 rounded-lg"
                    />
                    <h3 className="font-bold font-md pt-2">
                      Quality Control Photo
                    </h3>
                    <p className="text-sm">
                      The following image has been rated by our AI and computer
                      vision algorithm and we have made the following
                      conclusions.{" "}
                    </p>
                  </div>
                  <div className="min-h-24 w-full rounded-lg py-2 border-destructive border p-4">
                    <p className="font-bold text-destructive self-start text-md">
                      SRCR
                    </p>
                    <p className="text-xs ">
                      Suitable Return Confidence Rating
                    </p>
                    <div className="flex flex-col items-center justify-center w-full py-1">
                      <Progress value={randomVar! * 100} className="" />
                      <span className="text-xs text-destructive text-medium">
                        {" "}
                        {randomVar?.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
                <div
                  className="border p-4 rounded-lg"
                  onClick={() => setIRB(!IndividualReturnBox)}
                >
                  <DataTableDemo />
                </div>
              </div>

              <div className="space-y-2 border rounded-lg p-4 my-2">
                <h1 className="font-bold text-2xl">Analysis:</h1>
                <h2 className="font-bold">
                  Condition:{" "}
                  <span className="font-normal">{response?.condition}</span>
                </h2>
                <h2 className="font-bold">
                  Authenticity:{" "}
                  <span className="font-normal">{response?.authenticity}</span>
                </h2>
                <h2 className="font-bold">
                  Packaging and Accesories:{" "}
                  <span className="font-normal">
                    {response?.packaging_and_accessories}
                  </span>
                </h2>
                <h2 className="font-bold">
                  Damage :{" "}
                  <span className="font-normal">{response?.damage}</span>
                </h2>
              </div>
            </div>
          </div>
          <div className="my-4">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIRB(!IndividualItem)}
            >
              Back
            </Button>
          </div>
        </>
      )}
    </>
  );
};

export default IndividualItem;
