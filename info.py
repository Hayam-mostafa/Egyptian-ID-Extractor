from typing import Dict

GOVERNORATES = {
    '01': 'Cairo',
    '02': 'Alexandria',
    '03': 'Port Said',
    '04': 'Suez',
    '11': 'Damietta',
    '12': 'Dakahlia',
    '13': 'Ash Sharqia',
    '14': 'Kaliobeya',
    '15': 'Kafr El Sheikh',
    '16': 'Gharbia',
    '17': 'Monoufia',
    '18': 'El Beheira',
    '19': 'Ismailia',
    '21': 'Giza',
    '22': 'Beni Suef',
    '23': 'Fayoum',
    '24': 'El Menia',
    '25': 'Assiut',
    '26': 'Sohag',
    '27': 'Qena',
    '28': 'Aswan',
    '29': 'Luxor',
    '31': 'Red Sea',
    '32': 'New Valley',
    '33': 'Matrouh',
    '34': 'North Sinai',
    '35': 'South Sinai',
    '88': 'Foreign'
}


def parse_egyptian_id(id_number):

    if not id_number or len(id_number) != 14 or not id_number.isdigit():
        return None

    try:
        century_digit = int(id_number[0])
        year = int(id_number[1:3])
        month = int(id_number[3:5])
        day = int(id_number[5:7])
        governorate_code = id_number[7:9]
        gender_code = int(id_number[12])

        if century_digit == 2:
            full_year = 1900 + year
        elif century_digit == 3:
            full_year = 2000 + year
        else:
            return None

        return {
            "full_year": full_year,
            "month": month,
            "day": day,
            "governorate_code": governorate_code,
            "birth_date": f"{full_year:04d}-{month:02d}-{day:02d}",
            "governorate": GOVERNORATES.get(governorate_code, "Unknown"),
            "gender": "Male" if gender_code % 2 else "Female"
        }

    except (ValueError, IndexError):
        return None

