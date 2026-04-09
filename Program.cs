using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ML_KV
{
    
        //описываем структуру данных что подаем на вход 
        public class HouseDate
        {
            [LoadColumn(0)]public float Size { get; set; } //площадь квадратных метров
            [LoadColumn(1)]public float Price { get; set; } //то, что учимся предсказывать
        }
        //описываем структуру предсказания (что получим на выходе)
        public class Prediction
        {
            [ColumnName("Score")]public float Price { get; set; } //цена исходя из предсказания
        }
        internal class Program
        {
        static void Main(string[] args)
        {
            var context = new MLContext();//Главный объект библиотеки ML.NET, двигатель нашего процесса
            //данные для обучения (мини база знаний:площадь и цена в миллион)
            var data = new[]
            {
                new HouseDate{Size=30,Price=3.0f},//площадь 30 квадратных метров, цена 3.000.000 и т.д.
                new HouseDate{Size=34,Price=4.0f},
                new HouseDate{Size=60,Price=6.0f},
                new HouseDate{Size=80,Price=8.0f},
                new HouseDate{Size=100,Price=10.0f},
            };
            //загружаем данные в паять 
            var trainingData = context.Data.LoadFromEnumerable(data);
            // говорим модели машинного обучения что Size это признак для расчета стоимости, а прайс это цель подучения
            var pipeline = context.Transforms.Concatenate("Features", "Size").Append(context.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));
            var model = pipeline.Fit(trainingData); //запускаем процесс оьучения модели
            //интерактив с пользователем
            Console.WriteLine("---Нейросеть-риэлтор готов к работе---");
            while (true)
            {
                Console.WriteLine("\nВведите площадь квартиры (кв.м.) или 0 для вывода");
                string input=Console.ReadLine();
                if (float.TryParse(input, out float userSize) && userSize > 0)
                {
                    //создаем функцию предсказания на условии обученной модели 
                    var predictionEngine = context.Model.CreatePredictionEngine<HouseDate, Prediction>(model);
                    // делаем прогноз для введеного числа 
                    var result = predictionEngine.Predict(new HouseDate { Size = userSize });
                    Console.WriteLine($"Прогноз нейросети: такая квартира стоит примерно {result.Price:F2} млн.");
                }
                else break;
            }

        }
    }
}
    