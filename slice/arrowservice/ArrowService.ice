// ArrowService.ice — Ice IDL for the casmservice Arrow data plane.
//
// ArrowServer stores and serves Apache Arrow tables serialized as Arrow IPC
// file-format bytes.  Clients upload tables by name and retrieve them by name.
// The server is intentionally schema-agnostic: schema information is encoded
// inside the IPC bytes and exposed separately via getTableSchema.

#pragma once

module arrowservice
{
    // Raw bytes sequence used to carry Arrow IPC payloads.
    sequence<byte> ByteSeq;

    // A serialized Arrow table in IPC file format.
    // `data` holds the raw bytes produced by pyarrow.ipc.new_file.
    struct ArrowTable
    {
        ByteSeq data;
    };

    // Sequence of table names returned by listTableNames.
    sequence<string> StringSeq;

    // Raised when a requested table does not exist on the server.
    exception TableNotFound
    {
        string tableName;
    };

    interface ArrowServer
    {
        // Upload (or replace) a named Arrow table.
        void uploadTable(string tableName, ArrowTable table);

        // Retrieve a previously uploaded table by name.
        // Throws TableNotFound if the name is unknown.
        ArrowTable getTable(string tableName) throws TableNotFound;

        // Return the Arrow schema of a table as a human-readable string.
        // Throws TableNotFound if the name is unknown.
        string getTableSchema(string tableName) throws TableNotFound;

        // Return the names of all currently stored tables.
        StringSeq listTableNames();
    };
};
