namespace BingoScanner.Shared
{
    public record Plate(Guid Id, int?[][] Cells)
    {
        public int ScoreAgainst(HashSet<int> called)
        {
            var s = 0;
            for (int r=0;r<Cells.Length;r++)
                for (int c=0;c<Cells[r].Length;c++)
                    if (Cells[r][c].HasValue && called.Contains(Cells[r][c]!.Value)) s++;
            return s;
        }
    }
}