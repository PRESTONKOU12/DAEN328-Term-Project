import requests
import pandas as pd
import re

# List of official Chicago ZCTAs (ZIP Code Tabulation Areas)
CHICAGO_ZIPS = [
    "60601", "60602", "60603", "60604", "60605", "60606", "60607", "60608", "60609", "60610",
    "60611", "60612", "60613", "60614", "60615", "60616", "60617", "60618", "60619", "60620",
    "60621", "60622", "60623", "60624", "60625", "60626", "60627", "60628", "60629", "60630", 
    "60631", "60632", "60633", "60634", "60635", "60636", "60637", "60638", "60639", "60640",
    "60641", "60642", "60643", "60644", "60645", "60646", "60647", "60649", "60651", "60652", 
    "60653", "60654", "60655", "60656", "60657", "60659", "60660", "60661", "60707", "60827"
]

# 60658 are not official Chicago ZCTAs but are often associated with the city.

def get_chicago_data(zip_list, release="acs2019_5yr"):
    all_results = []

    match = re.fullmatch(r"acs(\d{4})_5yr", release)
    if not match:
        raise ValueError(
            "Release must look like 'acsYYYY_5yr' (example: acs2019_5yr for 2015-2019 data)."
        )

    year = match.group(1)
    variables = [
        "B01002_001E",  # median age
        "B19013_001E",  # median household income
        "B02001_002E",  # white alone
        "B02001_003E",  # black alone
        "B02001_005E",  # asian alone
        "B02001_007E",  # some other race alone
        "B12001_001E",  # total marital-status universe
        "B12001_004E",  # male now married
        "B12001_013E",  # female now married
        "B13002_002E",  # women who had a birth in last 12 months
        "B15003_001E",  # education total
        "B15003_022E",  # bachelor's
        "B15003_023E",  # master's
        "B15003_024E",  # professional school
        "B15003_025E",  # doctorate
    ]
    
    for z in zip_list:
        if z not in CHICAGO_ZIPS:
            print(f"Skipping {z}: Not a recognized Chicago city ZIP.")
            continue
        
        try:
            url = f"https://api.census.gov/data/{year}/acs/acs5"
            params = {
                "get": ",".join(variables),
                "for": f"zip code tabulation area:{z}",
                "in": "state:17",
            }

            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, list) or len(payload) < 2:
                raise ValueError(f"Unexpected API response for {z}: {payload}")

            headers = payload[0]
            values = payload[1]
            row_data = dict(zip(headers, values))

            def to_num(key):
                value = row_data.get(key)
                if value in (None, "", "null"):
                    return None
                return float(value)

            row = {'zip_code': z, 'release': release}
            
            # Median Age
            row['MedianAge'] = to_num('B01002_001E')
            
            # Median HH Income
            row['MedianIncome'] = to_num('B19013_001E')
            
            # Predominant Race (Logic: Find highest count)
            races = {
                'White': to_num('B02001_002E') or 0,
                'Black': to_num('B02001_003E') or 0,
                'Asian': to_num('B02001_005E') or 0,
                'Other': to_num('B02001_007E') or 0,
            }
            # Simplistic predominant race check
            row['PredRace'] = max(races, key=races.get)
            
            # Marital Status (% Currently Married)
            total_mar = to_num('B12001_001E')
            married = (to_num('B12001_004E') or 0) + (to_num('B12001_013E') or 0)
            row['PctMarried'] = round((married / total_mar) * 100, 2) if total_mar else None
            
            # Fertility (Births in last 12 months)
            row['BirthRate'] = to_num('B13002_002E')
            
            # Education (% Bachelor's or Higher)
            edu_total = to_num('B15003_001E')
            bach_plus = sum((to_num(code) or 0) for code in ['B15003_022E', 'B15003_023E', 'B15003_024E', 'B15003_025E'])
            row['EduRate'] = round((bach_plus / edu_total) * 100, 2) if edu_total else None
            
            all_results.append(row)
            print(f"Successfully processed {z}")
            
        except Exception as e:
            print(f"Could not process {z}: {e}")
            
    return pd.DataFrame(all_results)

df = get_chicago_data(CHICAGO_ZIPS)

# save to CSV locally to C:\Users\ealss\DAEN_328
df.to_csv(r'C:\Users\ealss\DAEN_328\chicago_census_data.csv', index=False)


