import requests

def stocks_name():
    url = "https://nsearchives.nseindia.com/content/fo/qtyfreeze.csv"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open("stocks_name.csv", "wb") as file:
            file.write(response.content)
        print("CSV file downloaded successfully as 'StocksName.csv'")
        # insert_stock_name()
        
    else:
        print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

stocks_name()
