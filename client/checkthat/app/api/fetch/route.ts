import dbConnect from "@/lib/dbConnect";
import { MongoClient } from "mongodb";
import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest){
    await dbConnect();
    

    const client = await MongoClient.connect(process.env.MONGODB_URI!);
    const db = client.db('hackdavis2024');
    const collection = db.collection('python');

    const document = await collection.findOneAndUpdate(
        { used: false }, // filter to find unused document
        { $set: { used: true } }, // update the "used" field to true
        { sort: { _id: -1 }, returnDocument: 'after' } // return the updated document
    );

    console.log(document)

    client.close(); 

    if (document) {
        return NextResponse.json(document, {status: 200});
    } else {
        return NextResponse.json({ message: 'No documents found' }, {status: 400});
    }

  
}