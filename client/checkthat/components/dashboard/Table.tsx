"use client";

import * as React from "react";
import {
  CaretSortIcon,
  ChevronDownIcon,
  DotsHorizontalIcon,
} from "@radix-ui/react-icons";
import {
  ColumnDef,
  ColumnFiltersState,
  SortingState,
  VisibilityState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import Link from "next/link";
import IndividualItem from "./IndividualItem";

const data: Payment[] = [
  {
    id: "m5gr84i9",
    date: new Date("April 27, 2024"),
    uuid: "e219e1d72955409a3da95e24",
    status: "success",
  },
  {
    id: "3u1reuv4",
    date: new Date("April 27, 2024"),
    uuid: "f7e7439cab03f561476b724f",
    status: "success",
  },
  {
    id: "derv1ws0",
    date: new Date("April 27, 2024"),
    uuid: "1377d60a24792c02e7cb4e1b",
    status: "failed",
  },
  {
    id: "5kma53ae",
    date: new Date("April 27, 2024"),
    uuid: "34a81ba8e525e3674c3e0266",
    status: "success",
  },
  {
    id: "bhqecj4p",
    date: new Date("April 28, 2024"),
    uuid: "e0114e38991c5d3822ee868e",
    status: "processing",
  },
];

export type Payment = {
  id: string;
  date: Date;
  uuid: string;
  status: "pending" | "processing" | "success" | "failed";
};

export const columns: ColumnDef<Payment>[] = [
  {
    id: "select",
    header: ({ table }) => (
      <Checkbox
        checked={
          table.getIsAllPageRowsSelected() ||
          (table.getIsSomePageRowsSelected() && "indeterminate")
        }
        onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
        aria-label="Select all"
      />
    ),
    cell: ({ row }) => (
      <Checkbox
        checked={row.getIsSelected()}
        onCheckedChange={(value) => row.toggleSelected(!!value)}
        aria-label="Select row"
      />
    ),
    enableSorting: false,
    enableHiding: false,
  },

  {
    accessorKey: "uuid",
    header: () => {
      return <div>Identification</div>;
    },
    cell: ({ row }) => <div className="lowercase">{row.getValue("uuid")}</div>,
  },
  {
    accessorKey: "status",
    header: "Returnable",
    cell: ({ row }) => (
      <div
        className={
          "capitalize " +
          (row.getValue("status") === "success"
            ? "text-green-400"
            : row.getValue("status") === "failed"
            ? "text-destructive"
            : "text-yellow-400")
        }
      >
        {row.getValue("status")}
      </div>
    ),
  },
  {
    accessorKey: "date",
    header: ({ column }) => (
      <div className="text-right">
        <Button
          variant="ghost"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Return Requested
          <CaretSortIcon className="ml-2 h-4 w-4" />
        </Button>
      </div>
    ),
    cell: ({ row }) => {
      const date: Date = row.getValue("date");

      // Format the amount as a dollar amount
      const formatted = new Intl.DateTimeFormat("en-US").format(date);

      return <div className="text-right font-medium">{formatted}</div>;
    },
  },
  {
    id: "actions",
    enableHiding: false,
    cell: ({ row }) => {
      const payment = row.original;

      return <DropdownMenu></DropdownMenu>;
    },
  },
];

export function DataTableDemo() {
  const [IndividualReturnBox, setIRB] = React.useState<Boolean>(false);
  const [selectedUUID, setUUID] = React.useState<string>("");
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    []
  );
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({});
  const [rowSelection, setRowSelection] = React.useState({});

  const table = useReactTable({
    data,
    columns,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    state: {
      sorting,
      columnFilters,
      columnVisibility,
      rowSelection,
    },
  });

  const handleClick = (uuid: string) => {
    setIRB(!IndividualReturnBox);
    setUUID(uuid);
  };

  return (
    <div className="w-full">
      {IndividualReturnBox ? (
        <IndividualItem
          uuid={selectedUUID}
          IndividualReturnBox={IndividualReturnBox}
          setIRB={setIRB}
        />
      ) : (
        <>
          <h2 className="font-semibold text-2xl">Returns</h2>
          <div className="flex items-center py-4">
            <Input
              placeholder="Filter ids..."
              value={
                (table.getColumn("uuid")?.getFilterValue() as string) ?? ""
              }
              onChange={(event) =>
                table.getColumn("uuid")?.setFilterValue(event.target.value)
              }
              className="max-w-sm"
            />
            <DropdownMenu></DropdownMenu>
          </div>
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow key={headerGroup.id}>
                    {headerGroup.headers.map((header) => {
                      return (
                        <TableHead key={header.id}>
                          {header.isPlaceholder
                            ? null
                            : flexRender(
                                header.column.columnDef.header,
                                header.getContext()
                              )}
                        </TableHead>
                      );
                    })}
                  </TableRow>
                ))}
              </TableHeader>
              <TableBody>
                {table.getRowModel().rows?.length ? (
                  table.getRowModel().rows.map((row) => (
                    <TableRow
                      key={row.id}
                      data-state={row.getIsSelected() && "selected"}
                      className="hover:cursor-pointer"
                    >
                      {row.getVisibleCells().map((cell) => (
                        <TableCell
                          key={cell.id}
                          onClick={() => handleClick(row.original.uuid)}
                        >
                          {flexRender(
                            cell.column.columnDef.cell,
                            cell.getContext()
                          )}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell
                      colSpan={columns.length}
                      className="h-24 text-center"
                    >
                      No results.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
          <div className="flex items-center justify-end space-x-2 py-4">
            <div className="flex-1 text-sm text-muted-foreground">
              {table.getFilteredSelectedRowModel().rows.length} of{" "}
              {table.getFilteredRowModel().rows.length} row(s) selected.
            </div>
            <div className="space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => table.previousPage()}
                disabled={!table.getCanPreviousPage()}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => table.nextPage()}
                disabled={!table.getCanNextPage()}
              >
                Next
              </Button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
