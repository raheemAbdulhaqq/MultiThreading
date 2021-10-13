using RestSharp;
using System;
using System.Collections.Generic;

namespace RESTSharp.Consume
{
    class Program
    {
        private static RestClient client = new
        RestClient("http://localhost:45767/api/");
        static void Main(string[] args)
        {
            RestRequest request = new RestRequest("Default", Method.GET);
            IRestResponse<List<string>> response = client.Execute<List<string>>(request);
            Console.WriteLine("{0}", response.Data);
            Console.ReadKey();

            //RestRequest request = new RestRequest("Default", Method.POST);
            //request.AddJsonBody("Robert Michael");
            //var response = client.Execute(request);
        }
    }
}
