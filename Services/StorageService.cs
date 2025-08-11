using System.Text.Json;
using BingoScanner.Shared;
using Microsoft.JSInterop;

namespace BingoScanner.Services
{
    public class StorageService
    {
        private readonly IJSRuntime _js;
        public StorageService(IJSRuntime js) => _js = js;

        public async ValueTask SavePlatesAsync(IEnumerable<Plate> plates)
        {
            var json = JsonSerializer.Serialize(plates);
            await _js.InvokeVoidAsync("localStorage.setItem", "plates", json);
        }

        public async Task<List<Plate>> LoadPlatesAsync()
        {
            var json = await _js.InvokeAsync<string>("localStorage.getItem", "plates");
            if (string.IsNullOrWhiteSpace(json)) return new();
            return JsonSerializer.Deserialize<List<Plate>>(json) ?? new();
        }
    }
    public class BingoPlateGenerator
    {
        public List<int?[][]> GenerateRandomPlates(int count = 30)
        {
            var plates = new List<int?[][]>();
            var rand = new Random();

            for (int i = 0; i < count; i++)
            {
                var plate = new int?[3][];
                for (int r = 0; r < 3; r++)
                    plate[r] = new int?[9];

                for (int col = 0; col < 9; col++)
                {
                    int start = col * 10 + 1;
                    int end = (col == 8) ? 90 : col * 10 + 10;
                    var nums = Enumerable.Range(start, end - start + 1)
                        .OrderBy(_ => rand.Next())
                        .Take(3)
                        .ToList();
                    nums.Sort();
                    for (int r = 0; r < 3; r++)
                        plate[r][col] = nums[r];
                }

                for (int r = 0; r < 3; r++)
                {
                    var blanks = Enumerable.Range(0, 9)
                        .OrderBy(_ => rand.Next())
                        .Take(4)
                        .ToList();
                    foreach (var b in blanks)
                        plate[r][b] = null;
                }

                plates.Add(plate);
            }

            return plates;
        }
    }
}