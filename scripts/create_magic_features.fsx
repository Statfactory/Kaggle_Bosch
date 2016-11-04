#r "Xi.dll"
open System
open System.IO
open Xi
open Xi.DataReader
open System.IO.Compression

//create magic features and import all data into Xi format
let sourceDir = __SOURCE_DIRECTORY__
let trainNumPath = Path.Combine(sourceDir, "train_numeric.csv")
let trainDatePath = Path.Combine(sourceDir, "train_date.csv")
let trainCatPath = Path.Combine(sourceDir, "train_categorical.csv")
let testNumPath = Path.Combine(sourceDir, "test_numeric.csv")
let testDatePath = Path.Combine(sourceDir, "test_date.csv")
let testCatPath = Path.Combine(sourceDir, "test_categorical.csv")
let trainMagicPath = Path.Combine(sourceDir, "train_magic.csv")
let testMagicPath = Path.Combine(sourceDir, "test_magic.csv")

type Rec =
    {
     Id : int
     Response : int option
     NumFeatures : string[]
     StartDate : float32
     EndDate : float32
     StationIds : Set<int>
     StationPath : string
     LinePath : string
     StationStart : float32[]
     StationEnd : float32[]
    }

type SeriesStats =
    {
     StartDate : float32
     MinEndDate : float32
     MaxEndDate : float32
     MinTime : float32
     MaxTime : float32
     StationIds : Set<int>
     SeriesLen : int
     SeriesTrainLen : int
     SeriesTestLen : int
     ResponseSum : int
     StationPathsSame : bool
     EndDatesSame : bool
     NumAllSame : bool
    }

let stationCount = 52

let parse (v : string) =
    match Single.TryParse(v) with
        | true, v  -> v
        | _ -> Single.NaN

let getRec (dateHeaders : string[]) (responseColIndex : int option) (idColIndex : int) ((numRow, dateRow) : string * string) =
    let numRow = numRow.Split([|','|])
    let dateRow = dateRow.Split([|','|]) |> Array.map parse
    let dates =
        seq{0..dateRow.Length - 1} |> Seq.filter (fun i -> i <> idColIndex && not <| Single.IsNaN(dateRow.[i]))
                                   |> Seq.map (fun i -> dateRow.[i])
    
    let startDate = if dates |> Seq.isEmpty then Single.NaN else dates |> Seq.min |> float32
    let endDate = if dates |> Seq.isEmpty then Single.NaN else dates |> Seq.max |> float32
    let stationIds = seq{0..dateRow.Length - 1} |> Seq.filter (fun i -> i <> idColIndex && not <| Single.IsNaN(dateRow.[i]))
                                                |> Seq.map (fun i -> dateHeaders.[i].Split([|'_'|]).[1].TrimStart([|'S'|]) |> int)
                                                |> Set.ofSeq
    let numFeatures = seq{0..numRow.Length - 1} |> Seq.filter (fun i -> i <> idColIndex && (match responseColIndex with | Some(index) -> i <> index | None -> true))
                                                |> Seq.map (fun i -> numRow.[i])
                                                |> Seq.toArray
    let stationPath = seq{0..dateRow.Length - 1} |> Seq.filter (fun i -> i <> idColIndex && not <| Single.IsNaN(dateRow.[i]))
                                                 |> Seq.map (fun i -> dateHeaders.[i].Split([|'_'|]).[1])
                                                 |> Seq.distinct
                                                 |> String.concat String.Empty
    let linePath = seq{0..dateRow.Length - 1} |> Seq.filter (fun i -> i <> idColIndex && not <| Single.IsNaN(dateRow.[i]))
                                              |> Seq.map (fun i -> dateHeaders.[i].Split([|'_'|]).[0])
                                              |> Seq.distinct
                                              |> String.concat String.Empty

    let stationStartEnd = seq{ 0..dateRow.Length - 1} |> Seq.filter (fun i -> i <> idColIndex)
                                                      |> Seq.map (fun i -> dateHeaders.[i].Split([|'_'|]).[1].TrimStart([|'S'|]) |> int, dateRow.[i])
                                                      |> Seq.groupBy fst
                                                      |> Seq.map (fun (stationId, dates) -> if dates |> Seq.isEmpty then
                                                                                                stationId, Single.NaN, Single.NaN
                                                                                            else
                                                                                                stationId, (dates |> Seq.map snd |> Seq.min), (dates |> Seq.map snd |> Seq.max)
                                                                 )
                                                      |> Seq.sortBy (fun (x,y,z) -> x)
                                                      |> Seq.map (fun (x,y,z) -> y,z)
                                                      |> Seq.toArray
                                                  

    {
     Id = numRow.[idColIndex] |> int
     Response = responseColIndex |> Option.map (fun colIndex -> numRow.[colIndex] |> int)
     StartDate = startDate
     EndDate = endDate
     StationIds = stationIds
     NumFeatures = numFeatures
     StationPath = stationPath
     LinePath = linePath
     StationStart = stationStartEnd |> Array.map fst
     StationEnd = stationStartEnd |> Array.map snd
    }

let rec merge (trainRows : DataReader<Rec>) (testRows : DataReader<Rec>) =
    match trainRows |> DataReader.tryRead, testRows |> DataReader.tryRead with
        | (Some(trainRec), trainTail), (Some(testRec), testTail) ->
            if trainRec.Id < testRec.Id then
                Cons(fun () -> (trainRec, true), merge trainTail (Cons(fun () -> testRec, testTail)))
            elif testRec.Id < trainRec.Id then
                Cons(fun () -> (testRec, false), merge (Cons(fun () -> trainRec, trainTail)) testTail)
            else raise (new InvalidOperationException("Ids cannot be equal in train and test"))
        | (None, _), (Some(testRec), testTail) -> 
            Cons(fun () -> testRec, testTail) |> DataReader.map (fun row -> row, false)
        | (Some(trainRec), trainTail), (None,_) ->
            Cons(fun () -> trainRec, trainTail) |> DataReader.map (fun row -> row, true)
        | (None, _), (None, _) -> Nil

let rec getSeries (acc : (Rec * bool) list) (rows : DataReader<Rec * bool>)  =
    match acc with
        | [] -> 
            match rows |> DataReader.tryRead with
                | (Some(row)), tail -> getSeries [row] tail
                | None, _ -> Nil
        | (headRec, _) :: _ -> 
            match rows |> DataReader.tryRead with
                | Some((r, isTrain)), tail -> 
                    if r.StartDate = headRec.StartDate && r.Id = headRec.Id + 1 then
                        getSeries ((r, isTrain) :: acc) tail
                    else 
                        Cons(fun () -> (acc |> List.rev), getSeries [(r, isTrain)] tail)
                | None, _ ->
                    Cons(fun () -> (acc |> List.rev), Nil)

let getSeriesStats (recs : (Rec * bool) list) =
    let nonNaEndDates = recs |> List.filter (fun r -> not <| Single.IsNaN(fst(r).EndDate))
    let minEndDate = if nonNaEndDates |> List.isEmpty then Single.NaN else nonNaEndDates |> List.map (fun r -> fst(r).EndDate) |> List.min
    let maxEndDate = if nonNaEndDates |> List.isEmpty then Single.NaN else nonNaEndDates |> List.map (fun r -> fst(r).EndDate) |> List.max
    let stationIds = recs |> List.map (fun r -> fst(r).StationIds) |> List.reduce Set.union
    let stationPathsSame = recs |> List.map (fun r -> fst(r).StationIds) |> List.distinct |> List.length = 1
    let respSum = recs |> List.map (fun r -> match fst(r).Response with | Some(x) -> x | None -> 0) |> List.sum
    let numAllSame = recs |> List.map (fun r -> fst(r).NumFeatures) |> List.distinct |> List.length = 1
    let startDate = fst(recs.Head).StartDate
    let serLen = recs |> List.length
    let serTrainLen = recs |> List.filter snd |> List.length
    {
     StartDate = startDate
     MinEndDate = minEndDate
     MaxEndDate = maxEndDate
     StationIds = stationIds
     StationPathsSame = stationPathsSame
     EndDatesSame = minEndDate = maxEndDate
     ResponseSum = respSum
     SeriesLen = serLen
     SeriesTrainLen = serTrainLen
     SeriesTestLen = serLen - serTrainLen
     MinTime = minEndDate - startDate
     MaxTime = maxEndDate - startDate
     NumAllSame = numAllSame
    }

let getCsvLine (r : Rec) (stats : SeriesStats) (faultNotNaStart : (int*float32)[][]) (faultStartEnd : (int * float32 * float32) list) =
    let y =
        match r.Response with
            | Some(x) -> x
            | None -> 0
    let respSum = stats.ResponseSum - y
    let stationStartEndSame = 
        Array.init stationCount (fun i -> if not <| Single.IsNaN(r.StationStart.[i]) && not <| Single.IsNaN(r.StationEnd.[i]) then Some(r.StationStart.[i] = r.StationEnd.[i]) else None)
    let inFaultPeriod = 
        (faultStartEnd |> List.filter (fun (id, s, e) -> r.Id <> id && r.StartDate >= s && r.EndDate <= e) |> List.length) > 0
    let inFaultPeriodPartial = 
        (faultStartEnd |> List.filter (fun (id, s, e) -> r.Id <> id && (r.StartDate <= e && r.EndDate >= s)) |> List.length) > 0

    let faultFree = [|0..stationCount - 1|] |> Array.Parallel.map (fun i -> 
                                                                       let start = r.StationStart.[i]
                                                                       if Single.IsNaN(start) then Single.NaN
                                                                       else
                                                                           if faultNotNaStart.[i].Length = 0 then Single.NaN
                                                                           else
                                                                               if y = 0 then
                                                                                   faultNotNaStart.[i] |> Array.map (fun (id, x) -> System.Math.Abs(x - start))
                                                                                                       |> Array.min
                                                                               elif faultNotNaStart.[i].Length = 1 then Single.NaN
                                                                               else
                                                                                   faultNotNaStart.[i] |> Array.filter (fun (id, x) -> r.Id <> id)
                                                                                                       |> Array.map (fun (id, x) -> System.Math.Abs(x - start))
                                                                                                       |> Array.min
                                                                    )

    let faultFreeAll = let s = faultFree |> Seq.filter (fun x -> not <| Single.IsNaN(x)) in if s |> Seq.isEmpty then Single.NaN else s |> Seq.min
            
    let part1 = sprintf "%f,%f,%f,%f,%f,%d,%d,%d,%d,%b,%b,%b,%s,%s" stats.StartDate stats.MinEndDate stats.MaxEndDate
                         stats.MinTime stats.MaxTime respSum stats.SeriesLen stats.SeriesTrainLen
                         stats.SeriesTestLen stats.StationPathsSame stats.EndDatesSame stats.NumAllSame r.StationPath r.LinePath 
    let part2 = seq{0..stationCount - 1} |> Seq.map (fun i -> Set.contains i stats.StationIds) |> Seq.map (fun x -> x.ToString()) |> String.concat ","
    let part3 = faultFree|> Seq.map (fun x -> x.ToString()) |> String.concat ","
    let part4 = stationStartEndSame |> Array.map (fun x -> match x with | Some(v) -> v.ToString() | None -> String.Empty) |> String.concat ","
    sprintf "%s,%s,%s,%s,%f,%b,%b" part1 part2 part3 part4 faultFreeAll inFaultPeriod inFaultPeriodPartial


let getFaultDates (trainNumPath : string) (trainDatePath : string) =
    use trainNumSDataReader = new StreamReader(trainNumPath)
    use trainDateSDataReader = new StreamReader(trainDatePath)
    let trainNumHeaders, trainNumDataReader = match trainNumSDataReader |> DataReader.createCsvReader |> DataReader.tryRead with | Some(h), r -> h, r | _ -> raise (new ArgumentException("no headers"))
    let trainDateHeaders, trainDateDataReader = match trainDateSDataReader |> DataReader.createCsvReader |> DataReader.tryRead with | Some(h), r -> h, r | _ -> raise (new ArgumentException("no headers"))

    let trainDateHeaders = trainDateHeaders.Split([|','|]) 
    let trainNumHeaders = trainNumHeaders.Split([|','|]) 
    let trainRespColIndex = trainNumHeaders |> Array.tryFindIndex (fun h -> h = "Response")
    let trainRows = trainDateDataReader |> DataReader.zip trainNumDataReader |> DataReader.map (getRec trainDateHeaders trainRespColIndex 0) 
    let faultRows = trainRows |> DataReader.filter (fun r -> match r.Response with | Some(resp) when resp = 1 -> true | _ -> false)
    faultRows |> DataReader.toList
        
let createFeatures (trainNumPath : string) (trainDatePath : string) (testNumPath : string) (testDatePath : string)
                   (trainMagicPath : string) (testMagicPath : string) =

    let faultRows = getFaultDates trainNumPath trainDatePath 

    let faultNotNaStart = [|0..stationCount - 1|] |> Array.Parallel.map (fun stationId ->
                                                                              faultRows |> List.map (fun r -> r.Id, r.StationStart.[stationId]) |> List.filter (fun (id, start) -> not <| Single.IsNaN(start)) |> List.toArray
                                                                         )
    let faultStartEnd = faultRows |> List.map (fun r -> r.Id, r.StartDate, r.EndDate)

    printfn "finished getFaultDates"

    use trainNumSDataReader = new StreamReader(trainNumPath)
    use trainDateSDataReader = new StreamReader(trainDatePath)
    use testNumSDataReader = new StreamReader(testNumPath)
    use testDateSDataReader = new StreamReader(testDatePath)

    let trainNumHeaders, trainNumDataReader = match trainNumSDataReader |> DataReader.createCsvReader  |> DataReader.tryRead with | Some(h), r -> h, r | _ -> raise (new ArgumentException("no headers"))
    let trainDateHeaders, trainDateDataReader = match trainDateSDataReader |> DataReader.createCsvReader  |> DataReader.tryRead with | Some(h), r -> h, r | _ -> raise (new ArgumentException("no headers"))
    let testNumHeaders, testNumDataReader = match testNumSDataReader |> DataReader.createCsvReader  |> DataReader.tryRead with | Some(h), r -> h, r | _ -> raise (new ArgumentException("no headers"))
    let testDateHeaders, testDateDataReader = match testDateSDataReader |> DataReader.createCsvReader  |> DataReader.tryRead with | Some(h), r -> h, r | _ -> raise (new ArgumentException("no headers"))

    let trainDateHeaders = trainDateHeaders.Split([|','|]) 
    let trainNumHeaders = trainNumHeaders.Split([|','|]) 
    let testDateHeaders = testDateHeaders.Split([|','|]) 
    let testNumHeaders = testNumHeaders.Split([|','|]) 

    let trainRespColIndex = trainNumHeaders |> Array.tryFindIndex (fun h -> h = "Response")
    let trainRows = trainDateDataReader |> DataReader.zip trainNumDataReader |> DataReader.map (getRec trainDateHeaders trainRespColIndex 0) 
    let testRows = testDateDataReader |> DataReader.zip testNumDataReader |> DataReader.map (getRec testDateHeaders None 0) 
    let merged = merge trainRows testRows
    let series = merged |> getSeries []

    use trainMagicWriter = new StreamWriter(trainMagicPath)
    use testMagicWriter = new StreamWriter(testMagicPath)
    let headers1 = "StartDate,MinEndDate,MaxEndDate,MinTime,MaxTime,ResponseSum,SeriesLen,SeriesTrainLen,SeriesTestLen,StationPathsSame,EndDatesSame,NumAllSame,StationPath,LinePath"
    let headers2 = seq{0..stationCount - 1} |> Seq.map (sprintf "Station%d") |> String.concat ","
    let headers3 = seq{0..stationCount - 1} |> Seq.map (sprintf "MinFaultFree%d") |> String.concat ","
    let headers4 = seq{0..stationCount - 1} |> Seq.map (sprintf "StationStartEndSame%d") |> String.concat ","
    let headers = sprintf "%s,%s,%s,%s,FaultFreeAll,InFaultPeriod,InFaultPeriodPartial" headers1 headers2 headers3 headers4
    trainMagicWriter.WriteLine (headers)
    testMagicWriter.WriteLine (headers)
    
    series 
           |> DataReader.iteri (fun i s -> 
                                if i % 50000 = 0 then
                                    printfn "processed %d" i
                                let stats = s |> getSeriesStats
                                s |> List.iter (fun (r, isTrain) ->
                                                    let line = getCsvLine r stats faultNotNaStart faultStartEnd
                                                    if isTrain then
                                                        trainMagicWriter.WriteLine line
                                                    else
                                                        testMagicWriter.WriteLine line
                                                )
                            )

#time
createFeatures trainNumPath trainDatePath testNumPath testDatePath trainMagicPath testMagicPath

async
    {
     do! DataImport.importCsvAsync trainNumPath [|','|] [||] Int32.MaxValue 10000 0.1 true String.Empty CompressionLevel.NoCompression
     do! DataImport.importCsvAsync trainCatPath [|','|] [||] Int32.MaxValue 10000 0.1 true String.Empty CompressionLevel.NoCompression
     do! DataImport.importCsvAsync trainMagicPath [|','|] [||] Int32.MaxValue 10000 0.0 true String.Empty CompressionLevel.NoCompression
     do! DataImport.importCsvAsync testNumPath [|','|] [||] Int32.MaxValue 10000 0.1 true String.Empty CompressionLevel.NoCompression
     do! DataImport.importCsvAsync testCatPath [|','|] [||] Int32.MaxValue 10000 0.1 true String.Empty CompressionLevel.NoCompression
     do! DataImport.importCsvAsync testMagicPath [|','|] [||] Int32.MaxValue 10000 0.0 true String.Empty CompressionLevel.NoCompression
     return ()
    } |> Async.Start

