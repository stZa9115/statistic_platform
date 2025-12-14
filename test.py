from costomTools.santize import sanitize_filename
with open("./check/sanitize_info", "w", encoding="utf-8") as f:
    f.write(sanitize_filename("..\\some / unsafe\\path\\file name @2024!.xlsx"))