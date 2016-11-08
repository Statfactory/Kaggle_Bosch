#r "Xi.dll"
open System
open System.IO
open Xi
open Xi.DataReader
open System.IO.Compression

//create duplicate based features and import all data into Xi format
let sourceDir = __SOURCE_DIRECTORY__
let train_num = Path.Combine(sourceDir, "train_numeric.csv")
let train_date = Path.Combine(sourceDir, "train_date.csv")
let test_num = Path.Combine(sourceDir, "test_numeric.csv")
let test_date = Path.Combine(sourceDir, "test_date.csv")

let trainDupsCsv = Path.Combine(sourceDir, "train_dups.csv")
let testDupsCsv = Path.Combine(sourceDir, "test_dups.csv")

let trainNumSDataReader = new StreamReader(train_num)
let trainDateSDataReader = new StreamReader(train_date)
let testNumSDataReader = new StreamReader(test_num)
let testDateSDataReader = new StreamReader(test_date)

type Rec =
    { 
     Id : int
     Response : int option
     NumFeatures : string
     DateFeatures : string
    }

type HashRec =
    {
     Id : int
     Response : int option
     NumHash : int
     DateHash : int
     NumDateHash : int
    }

type DupSummary =
    {
     DupCount : int
     ConsecutiveIds : bool
     RespSum : int option
     TestCount : int
    }

let getHashRec (r : Rec) =
    {
     Id = r.Id
     Response = r.Response
     NumHash = r.NumFeatures.GetHashCode()
     DateHash = r.DateFeatures.GetHashCode()
     NumDateHash = ([r.NumFeatures; r.DateFeatures] |> String.concat ",").GetHashCode()
    }

let split (s : string) = s.Split([|','|])

#time
let trainNumReader = trainNumSDataReader |> DataReader.createCsvReader
let _, trainNumRows = match trainNumReader |> DataReader.tryRead with | Some(h), rows -> h, rows | _ -> raise (new InvalidOperationException())

let trainDateReader = trainDateSDataReader |> DataReader.createCsvReader
let _, trainDateRows = match trainDateReader |> DataReader.tryRead with | Some(h), rows -> h, rows | _ -> raise (new InvalidOperationException())

let testNumReader = testNumSDataReader |> DataReader.createCsvReader
let _, testNumRows = match testNumReader |> DataReader.tryRead with | Some(h), rows -> h, rows | _ -> raise (new InvalidOperationException())

let testDateReader = testDateSDataReader |> DataReader.createCsvReader
let _, testDateRows = match testDateReader |> DataReader.tryRead with | Some(h), rows -> h, rows | _ -> raise (new InvalidOperationException())

let getRec (hasResponse : bool) (numRow : string, dateRow : string)  =
    let numFields = numRow |> split
    let dateFields = dateRow |> split
    let numFeatures = Array.sub numFields 1 (numFields.Length - if hasResponse then 2 else 1) |> String.concat "," 
    let dateFeatures = Array.sub dateFields 1 (dateFields.Length - 1) |> String.concat "," 
    {Id = numFields.[0] |> int; Response = if hasResponse then numFields.[numFields.Length - 1] |> int |> Some else None
     NumFeatures = numFeatures; DateFeatures = dateFeatures}

let trainRecs = DataReader.zip trainNumRows trainDateRows |> DataReader.map (getRec true)
let testRecs = DataReader.zip testNumRows testDateRows |> DataReader.map (getRec false)
let trainTestRecs = [trainRecs; testRecs] |> DataReader.concat
                                          |> DataReader.map getHashRec
                                          |> DataReader.toList

let getDups (getHash : HashRec -> int) (hashRecs : HashRec list)  =
    hashRecs |> Seq.groupBy getHash
             |> Seq.filter (fun (hash, g) -> g |> Seq.length > 1)
             |> Seq.toList
             |> List.map (fun (hash, g) ->
                                 let g = g |> Seq.toList
                                 let ids = g |> List.map (fun r -> r.Id)
                                 let minId = ids |> List.min
                                 let maxId = ids |> List.max
                                 let consecutiveIds = (maxId = minId + ids.Length - 1)  && [minId..maxId] = (ids |> List.sort)
                                 let dupCount = g.Length
                                 let respSum = g |> List.fold (fun s r -> 
                                                                 match s, r.Response with
                                                                     | Some(x), Some(y) -> (x + y) |> Some
                                                                     | Some(x), None -> s
                                                                     | None, Some(y) -> r.Response
                                                                     | None, None -> None
                                                             ) None
                                 let testCount = g |> Seq.filter (fun hr -> hr.Response.IsNone) |> Seq.length
                                 let dupSummary = {DupCount = dupCount; ConsecutiveIds = consecutiveIds; RespSum = respSum; TestCount = testCount}
                                 hash, dupSummary
                         ) |> Map.ofList

let numDups = trainTestRecs |> getDups (fun r -> r.NumHash)
let dateDups = trainTestRecs |> getDups (fun r -> r.DateHash)
let numDateDups = trainTestRecs |> getDups (fun r -> r.NumDateHash)

let trainDupsWriter = new StreamWriter(trainDupsCsv)
let testDupsWriter = new StreamWriter(testDupsCsv)

let headers = "NumDupCount,NumDupRespSum,NumDupConsecutiveIds,DateDupCount,DateDupRespSum,DateDupConsecutiveIds,NumDateDupCount,NumDateDupRespSum,NumDateDupConsecutiveIds"
trainDupsWriter.WriteLine headers
testDupsWriter.WriteLine headers

type Option<'a> with
    member this.AsString = match this with | Some(v) -> v.ToString() | None -> "."

let getSummary (r : HashRec) (dups : Map<int, DupSummary>) (getHash : HashRec -> int) =
    let hash = r |> getHash
    if dups.ContainsKey(hash) then
        let summary = dups.[hash]
        let resp = 
            match r.Response, summary.RespSum with
                | Some(x), Some(y) ->
                    if summary.DupCount - summary.TestCount = 1 then None
                    else (y - x) |> Some
                | None, Some(y) -> summary.RespSum
                | Some(x), None -> raise (new InvalidOperationException())
                | None, None -> None
        sprintf "%d,%s,%b" summary.DupCount resp.AsString summary.ConsecutiveIds
    else
        sprintf "1,.,."


trainTestRecs |> List.iter (fun hashRec ->
                                let numSummary = getSummary hashRec numDups (fun r -> r.NumHash)
                                let dateSummary = getSummary hashRec dateDups (fun r -> r.DateHash)
                                let numDateSummary = getSummary hashRec numDateDups (fun r -> r.NumDateHash)
                                let csvLine = [numSummary;dateSummary;numDateSummary] |> String.concat ","
                                match hashRec.Response with
                                    | Some(_) -> trainDupsWriter.WriteLine csvLine
                                    | None -> testDupsWriter.WriteLine csvLine
                           )


trainDupsWriter.Flush()
trainDupsWriter.Dispose()
testDupsWriter.Flush()
testDupsWriter.Dispose()

async
    {
     do! DataImport.importCsvAsync trainDupsCsv [|','|] [||] Int32.MaxValue 10000 0.0 true String.Empty CompressionLevel.NoCompression
     do! DataImport.importCsvAsync testDupsCsv [|','|] [||] Int32.MaxValue 10000 0.0 true String.Empty CompressionLevel.NoCompression
     return ()
    } |> Async.Start


