from rich.console import Console
from rich.table import Table
from varname import nameof

def table(
        data: list,
    ):
    table = Table(show_header=False)

    table.add_column(no_wrap=True)
    table.add_column()

    data = {nameof(d): d for d in data}

    for k, v in data.items():
        string = v.__repr__()
        if len(string) > 30:
            string = string[:200] + "..."
        name = nameof(k)
        table.add_row(f"[bold]{name}[/bold]", string, end_section=True)
        
    console = Console()
    console.print(table)


# # %%

# from rich.console import Console
# from rich.table import Table

# # Create a new table with two columns for the vertical layout
# vertical_table = Table(show_header=False)

# # Add two columns: one for the headers and one for the values
# vertical_table.add_column( no_wrap=True)
# vertical_table.add_column()


# # For each movie, add rows where each row represents a field (Released, Title, Box Office)
# movies = [
#     {"Released": "Dec 20, 2019", "Title": "Star Wars: The Rise of Skywalker", "Box Office": "$952,110,690"},
#     {"Released": "May 25, 2018", "Title": "Solo: A Star Wars Story", "Box Office": "$393,151,347"},
#     {"Released": "Dec 15, 2017", "Title": "Star Wars Ep. V111: The Last Jedi", "Box Office": "$1,332,539,889"},
#     {"Released": "Dec 16, 2016", "Title": "Rogue One: A Star Wars Story", "Box Office": "$1,332,439,889"},
# ]

# # Iterate over each movie to add its details in a vertical format
# for movie in movies:
#     vertical_table.add_row("[bold]Title[/bold]", movie["Title"], end_section=True)
#     vertical_table.add_row("Released", movie["Released"])
#     vertical_table.add_row("Box Office", movie["Box Office"], end_section=True)

# console = Console()
# console.print(vertical_table)

# # %%

