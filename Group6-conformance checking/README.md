# پروژه تحلیل انطباق فرآیند با PM4Py

این پروژه برای تحلیل انطباق فرآیند طراحی شده و با استفاده از مدل پتری نت (PNML) و لاگ رویداد (CSV)، انطباق هر case را با مدل بررسی می کند. الگوریتم اصلی مورد استفاده Token Replay از کتابخانه PM4Py است. علاوه بر آن، معیارهای پیشرفته انطباق (Fitness و Precision) نیز برای کل لاگ محاسبه و در خروجی JSON ذخیره می شوند.

## هدف پروژه
- بارگذاری مدل فرآیند (Petri Net)
- بارگذاری لاگ رویداد
- اجرای Token Replay
- جداسازی case های منطبق و غیرمنطبق
- محاسبه معیارهای پیشرفته انطباق (Fitness و Precision)
- ذخیره خروجی در فایل های CSV و JSON

## ورودی ها
- فایل مدل با فرمت PNML
- فایل لاگ رویداد CSV با ستون های:
  - case_id یا case:concept:name
  - activity یا concept:name
  - timestamp یا time:timestamp

## خروجی ها
- compliant_<model>_<log>.csv
- non_compliant_<model>_<log>.csv
- <model>_<log>_summary.json

## نسخه های تست شده
- pm4py 2.7.19.2
- pandas 2.3.1
- numpy 2.0.2
- matplotlib 3.9.4
- networkx 3.2.1
- graphviz 0.21
- Python 3.10

## نحوه اجرا
```python
pnml_file_path = "./data/heuristic_SP_DF_cleaned.pnml"
log_csv_file_path = "./data/SP_DF_cleaned.csv"
process_log(pnml_file_path, log_csv_file_path)
```



## توضیح فایل های CSV خروجی
- فایل `compliant_<model>_<log>.csv` شامل تمام رخدادهای مربوط به case هایی است که به طور کامل با مدل پتری نت منطبق بوده اند.
- فایل `non_compliant_<model>_<log>.csv` شامل تمام رخدادهای مربوط به case هایی است که حداقل یک انحراف از مدل فرآیند داشته اند.
- 
## نمونه خروجی JSON
```json
{
    "model_file": "heuristic_SP_DF_cleaned.pnml",
    "log_file": "SP_DF_cleaned.csv",
    "timestamp_key_used": "time:timestamp",
    "stats": {
        "total_cases": 100000,
        "compliant_cases": 47127,
        "non_compliant_cases": 52873,
        "compliant_percentage": 47.127,
        "non_compliant_percentage": 52.873000000000005
    },
    "advanced_conformance": {
        "fitness": {
            "perc_fit_traces": 47.127,
            "average_trace_fitness": 0.9272932457539153,
            "log_fitness": 0.9173290234278992,
            "percentage_of_fitting_traces": 47.127
        },
        "precision": 0.9082817362189379
    }
}
```
