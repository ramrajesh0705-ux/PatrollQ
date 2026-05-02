def select_features(df):
    return df[["Latitude", "Longitude","Hour", "Month","DayOfWeek","IsWeekend","CrimeSeverity"]]
