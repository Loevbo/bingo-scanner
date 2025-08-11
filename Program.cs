using BingoScanner;
using BingoScanner.Services;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");

builder.Services.AddScoped<StorageService>();
builder.Services.AddScoped<BingoPlateGenerator>();

await builder.Build().RunAsync();