using Microsoft.ML;
using ML_Classification_.DataModels;
using System;
using System.IO;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace ML_Classification_
{
    class Program
    {
        static readonly string _dataFilePath = Path.Combine(Environment.CurrentDirectory, "Data", "trainData.csv");
        static void Main(string[] args)
        {
            //Creating MLContxet model to be shared accross model building, validation and prediction process
            MLContext mlContext = new MLContext();

            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(_dataFilePath, hasHeader: true, separatorChar: ',', allowSparse: false);

            //Loading the data from csv files
            //Splitting the dataset into train/test sets
            TrainTestData trainTestData = LoadData(mlContext, testDataFraction: 0.25);

            //Creating data transformation pipeline which transforma the data a form acceptable by model
            //Returns an object of type IEstimator<ITransformer>
            var pipeline = ProcessData(mlContext);

            //passing the transformation pipeline and training dataset to crossvalidate and build the model
            //returns the model object of type ITransformer 
            var trainedModel = BuildAndTrain(mlContext, pipeline, trainTestData.TrainSet);

            mlContext.Model.Save(trainedModel, dataView.Schema, "Classification_Model.zip");
            //Sample datainput for predicrtion
            var Input1 = new ModelInput
            {
                skill_first = "Home Care",
                skill_second = "ICU",
                skill_third = "Empathy",
                skill_fourth = "GMP",
                skill_fifth = "Consultation"

            };

            //Sample datainput for predicrtion
            var Input2 = new ModelInput
            {
                skill_first = "Food Preparation",
                skill_second = "Nutrition Planning",
                skill_third = "Communication",
                skill_fourth = "Advising",
                skill_fifth = "Leadership"
            };



            //passing trained model and sample input data to make single prediction 
            var result = PredictSingleResult(mlContext, trainedModel, Input1);

            Console.WriteLine("================================= Single Prediction Result ===============================");
            // Evaluate(mlContext, pipeline, trainedModel,  mushroomTrainTestData.TestSet);
            Console.WriteLine($"Predicted Result: {result.Label}");

            Console.ReadKey();




        }

        public static TrainTestData LoadData(MLContext mlContext, double testDataFraction)
        {
            //Read data
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(_dataFilePath, hasHeader: true, separatorChar: ',', allowSparse: false);

            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: testDataFraction);

            return trainTestData;
        }

        public static IEstimator<ITransformer> ProcessData(MLContext mlContext)
        {
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(ModelInput.id))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "skill_first", outputColumnName: "skill_firstFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "skill_second", outputColumnName: "skill_secondFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "skill_third", outputColumnName: "skill_thirdFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "skill_fourth", outputColumnName: "skill_fourthFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "skill_fifth", outputColumnName: "skill_fifthFeaturized"))
                .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", inputColumnNames: new string[] { "skill_firstFeaturized",
                    "skill_secondFeaturized",
                    "skill_thirdFeaturized",
                    "skill_fourthFeaturized",
                    "skill_fifthFeaturized" }));
            return pipeline;
        }

        public static ITransformer BuildAndTrain(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView trainDataView)
        {
            //   PeekDataViewInConsole(mlContext, trainDataView, pipeline, 2);

            var trainPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", numberOfIterations: 10)))
                                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("=============== Starting 10 fold cross validation ===============");
            var crossValResults = mlContext.MulticlassClassification.CrossValidate(data: trainDataView, estimator: trainPipeline, numberOfFolds: 10, labelColumnName: "Label");

            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();


            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();


            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();


            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();


            //Console.WriteLine($"*************************************************************************************************************");
            //Console.WriteLine($"*       Metrics Classification model      ");
            //Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            //Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###} ");
            //Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###} ");
            //Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###} ");
            //Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###} ");
            //Console.WriteLine($"*************************************************************************************************************");


            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = trainPipeline.Fit(trainDataView);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;

        }

        public static ModelPrediction PredictSingleResult(MLContext mlContext, ITransformer model, ModelInput input)
        {

            //Creating the prediction engine which takes data model input and output
            var predictEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelPrediction>(model);

            var predOutput = predictEngine.Predict(input);

            return predOutput;


        }

    }
}
