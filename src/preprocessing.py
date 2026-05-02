import pandas as pd

def clean_data(df):
    """
    Robust cleaning for large, messy Chicago crime data
    """

    
    # Handle missing values
    df = df.dropna(subset=['Latitude', 'Longitude', 'Date'])
        
    df['Date'] = pd.to_datetime(df['Date'])
    # Extract temporal features
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
    # Time categories
    df['TimeOfDay'] = pd.cut(df['Hour'], 
                                bins=[0, 6, 12, 18, 24],
                                labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Severity scores
    severity_map = {

# Very High Severity (Violent crimes)
'HOMICIDE':5,
'HUMAN TRAFFICKING':5,
'CRIMINAL SEXUAL ASSAULT':5,
'KIDNAPPING':5,

# High Severity
'ROBBERY':4,
'ASSAULT':4,
'BATTERY':4,
'ARSON':4,
'WEAPONS VIOLATION':4,
'INTIMIDATION':4,
'STALKING':4,

# Medium Severity
'BURGLARY':3,
'MOTOR VEHICLE THEFT':3,
'NARCOTICS':3,
'OTHER NARCOTIC VIOLATION':3,
'SEX OFFENSE':3,
'OFFENSE INVOLVING CHILDREN':3,
'PROSTITUTION':3,
'INTERFERENCE WITH PUBLIC OFFICER':3,

# Low Severity (Property / regulatory crimes)
'THEFT':2,
'DECEPTIVE PRACTICE':2,
'CRIMINAL DAMAGE':2,
'CRIMINAL TRESPASS':2,
'PUBLIC PEACE VIOLATION':2,
'CONCEALED CARRY LICENSE VIOLATION':2,

# Minor violations
'LIQUOR LAW VIOLATION':1,
'PUBLIC INDECENCY':1,
'OBSCENITY':1,
'GAMBLING':1,
'OTHER OFFENSE':1,
'NON-CRIMINAL':1
}
    df['CrimeSeverity'] = df['Primary Type'].map(severity_map).fillna(3)
   
    df.drop(['ID','Case Number','Updated On'],axis=1,inplace=True)
    df = df.sample(n=500000,random_state=42)

    return df