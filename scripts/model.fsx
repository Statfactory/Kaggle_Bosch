#r "Xi.dll"
open System
open System.IO
open Xi
open Xi.ExplicitConversion
open Xi.StatModels
open Xi.Math
open Xi.BasicStats
open Xi.DataImport
open Xi.XiDataProvider

#time
type NumTrain = XiDataFrame< @"C:\Users\Adamm\Kaggle\Bosch\train_numeric.zip" >
let numTrain = new NumTrain()

type NumTest = XiDataFrame< @"C:\Users\Adamm\Kaggle\Bosch\test_numeric.zip" >
let numTest = new NumTest()

type CatTrain = XiDataFrame< @"C:\Users\Adamm\Kaggle\Bosch\train_categorical.zip" >
let catTrain = new CatTrain()

type CatTest = XiDataFrame< @"C:\Users\Adamm\Kaggle\Bosch\test_categorical.zip" >
let catTest = new CatTest()

type MagicTrain = XiDataFrame< @"C:\Users\Adamm\Kaggle\Bosch\train_magic.zip" >
let magicTrain = new MagicTrain()

type MagicTest = XiDataFrame< @"C:\Users\Adamm\Kaggle\Bosch\test_magic.zip" >
let magicTest = new MagicTest()

let resp = (numTrain.Response |>> float)

let numTrainFactors = numTrain.DataFrame.Factors |> List.filter (fun f -> f <> numTrain.Response)

let searchNumFactors = numTrainFactors |> List.map (fun x -> x.AsExpr)

let searchCatFactors = catTrain.DataFrame.Factors |> List.filter (fun f -> f.Cardinality > 1) |> List.map (fun x -> x.AsExpr)

let magicCovs = magicTrain.DataFrame.Covariates |> List.map (fun c -> Cut(c.AsExpr, use c = !!c in Vector.Unique c))
                                                |> List.filter (fun f -> f.Cardinality > 1)

let magicFactors = magicTrain.DataFrame.Factors |> List.filter (fun f -> f.Cardinality > 1) |> List.map (fun f -> f.AsExpr)


let obsFilterSeries = magicTrain.SeriesLen .> 1.0
let obsFilterNotSeries = magicTrain.SeriesLen .= 1.0

let hotFactorsSeriesTest = (magicCovs @ magicFactors @  searchNumFactors  @ searchCatFactors) |> List.map (fun f -> f.AsOneHots()) |> List.concat
                               |> List.map (fun f -> f, fisherExactTest f numTrain.Response.AsExpr (Some obsFilterSeries) 10000 4)
                               |> List.sortBy snd

let hotFactorsNotSeriesTest = (magicCovs @ magicFactors @  searchNumFactors  @ searchCatFactors) |> List.map (fun f -> f.AsOneHots()) |> List.concat
                                  |> List.map (fun f -> f, fisherExactTest f numTrain.Response.AsExpr (Some obsFilterNotSeries) 10000 4)
                                  |> List.sortBy snd

let hotFactorsSeries = hotFactorsSeriesTest |> List.filter (fun (f, t) -> t <= 0.05) |> List.map fst
let hotFactorsNotSeries = hotFactorsNotSeriesTest |> List.filter (fun (f, t) -> t <= 0.05) |> List.map fst


let lambda = 1.0
let gamma = 0.0
let minH = 3.0
let learnRate = 0.1

let trainFilter = new BoolCovariate(100u, resp.Length, 0.0, 0.8)

let xgSeries = ML.xgb resp (fun i -> hotFactorsSeries) (Some obsFilterSeries) (Some trainFilter.AsExpr) learnRate 30 lambda gamma minH (fun i -> 7) 10000 4

let xgNotSeries = ML.xgb resp (fun i -> hotFactorsNotSeries) (Some obsFilterNotSeries) (Some trainFilter.AsExpr) learnRate 30 lambda gamma minH (fun i -> 7) 10000 4

let seriesFilter :  BoolVector = !!obsFilterSeries.AsBoolCovariate
let notSeriesFilter :  BoolVector = !!obsFilterNotSeries.AsBoolCovariate
let r : Vector = !!resp.AsCovariate
let trainPredSeries = xgSeries.Predicted
let trainPredNotSeries = xgNotSeries.Predicted
let trainPred = iif seriesFilter trainPredSeries trainPredNotSeries |> eval
let bestMcc, bestMccCutoff = Metrics.bestMccCutoff trainPred r

let testframe = numTest.DataFrame + magicTest.DataFrame + catTest.DataFrame 
let testPredSeries = xgSeries.Predict testframe
let testPredNotSeries = xgNotSeries.Predict testframe
let testPred = iif (testPredSeries .= testPredSeries) testPredSeries testPredNotSeries |> eval

let testResponse = new Vector(testPred.Length, 0.0)
testResponse.[testPred .>= bestMccCutoff] <- !!1.0

let checkTestMean = mean testResponse
let responseCov = new Covariate("Response", !!testResponse)
Glm.toCsv [!!numTest.Id; !!responseCov] (Path.Combine(__SOURCE_DIRECTORY__, "result.csv"))



