import pandas as pd
from pyspark.sql.types import FloatType
from pyspark.sql.functions import abs, col, max, sum, when
from pyspark.sql import SparkSession
import logging

# Initialize Spark session
spark = SparkSession.builder.appName("asp-forecasting").getOrCreate()


def forecast_validation(
    model_validation_df,
    market_events_df: pd.DataFrame,
    champion_version: str,
    logger: logging.Logger,
):
    """
    Calculates the forecast confidence score and confidence category as High/Low.

    Args:
        model_validation_df (spark.DataFrame): DataFrame that contains model validation
        market_events_df (pd.DataFrame): DataFrame that contains market event features
        champion_version (str): version number of champion model.
        logger (logging.Logger): logger to log the information.

    Returns:
        DataFrame that contains forecast confidence score and confidence category.
    """
    child_logger = logger.getChild(
        "Calculating the forecast confidence score for all products"
    )
    market_events_df = spark.createDataFrame(market_events_df)

    # filter model validation table to consider only records of champion model.
    model_validation_df = model_validation_df.filter(
        model_validation_df["model_version_number"] == champion_version
    )

    # Select the maximum date from the 'QUARTER_START_DATE' column in the 'cms' df
    max_date = model_validation_df.select(max("QUARTER_START_DATE")).collect()[0][0]

    # Filter the 'validation' DataFrame to include only rows with the maximum date
    mdl_vldn_latest = model_validation_df.filter(
        model_validation_df.QUARTER_START_DATE == max_date
    )

    mdl_vldn_latest_df = mdl_vldn_latest.withColumn(
        "PRICE_CONFIDENCE",
        when(
            (col("QUARTER_START_DATE") == max_date)
            & (col("ASP_PRC_ACTL").cast(FloatType()) > 3),
            "High",
        ).otherwise("Low"),
    )

    model_validation_df = model_validation_df.withColumn(
        "SUCCESS_SCORE",
        100
        - (
            abs(
                (
                    col("ASP_PRC_FORECASTED").cast(FloatType())
                    - col("ASP_PRC_ACTL").cast(FloatType())
                )
            )
            * 100
            / col("ASP_PRC_ACTL").cast(FloatType())
        ),
    )

    # Group the data by J_CODE and calculate the average success score
    # The avg is calculated as the sum of success scores/num of forecast months
    # The average success score is then converted to a percentage format
    # A validation confidence column is added based on the average success score
    # If the avg success score is 0.8 or higher, the confidence is 'High', else 'Low'
    mdl_vldn_grouped = (
        model_validation_df.groupBy("J_CODE")
        .agg((sum("SUCCESS_SCORE") / 8.0).alias("AVG_SUCCESS_SCORE_PERCENT"))
        .withColumn(
            "VALIDATION_CONFIDENCE",
            when(col("AVG_SUCCESS_SCORE_PERCENT") >= 80, "High").otherwise("Low"),
        )
    )

    # Select the maximum date from the 'ASP_MTH' column in the 'Market Events' DataFrame
    max_date_market_events = market_events_df.select(max("ASP_MTH")).collect()[0][0]

    # Filter the 'cms' DataFrame to include only rows with the maximum date
    market_events_df = market_events_df.filter(
        market_events_df.ASP_MTH == max_date_market_events
    )

    # Classify market event confidence based on time conditions
    # related to competitive launches and loss of exclusivity (LOE)
    mrkt_ftr_classified = market_events_df.withColumn(
        "MARKET_EVENT_CONFIDENCE",
        when(
            # Low confidence if any event occurs within 12 months
            (col("TIME_SINCE_LAST_COMP_LAUNCH") <= 12)
            | (col("TIME_TO_NEXT_COMP_LAUNCH") <= 12)
            | (col("TIME_SINCE_LAST_LOE") <= 12)
            | (col("TIME_TO_NEXT_LOE") <= 12)
            | (col("TIME_SINCE_SAME_CLASS_LAUNCH") <= 12),
            "Low",
        ).otherwise(
            "High"
        ),  # High confidence for all other cases
    )

    # Join mdl_vldn_grouped with mdl_vldn_latest_df 
    # to include PRICE_CONFIDENCE based on J_CODE
    res1 = (
        mdl_vldn_grouped.select("J_CODE", "VALIDATION_CONFIDENCE")
        .distinct()
        .join(
            mdl_vldn_latest_df.select("J_CODE", "PRICE_CONFIDENCE"),
            on="J_CODE",
            how="left",
        )
    )

    # Further join with mrkt_ftr_classified to include MARKET_EVENT_CONFIDENCE
    res2 = res1.join(
        mrkt_ftr_classified.select("J_CODE", "MARKET_EVENT_CONFIDENCE"),
        on="J_CODE",
        how="left",
    )

    # Calculate individual scores for Validation, Price, 
    # and Market Event based on their confidence levels
    res3 = (
        res2.withColumn(
            "VALIDATION_SCORE",
            when(col("VALIDATION_CONFIDENCE") == "High", 1).otherwise(0),
        )
        .withColumn(
            "PRICE_SCORE", when(col("PRICE_CONFIDENCE") == "High", 1).otherwise(0)
        )
        .withColumn(
            "MARKET_EVENT_SCORE",
            when(col("MARKET_EVENT_CONFIDENCE") == "High", 1).otherwise(0),
        )
    )

    # Compute the Final Score by weighting Validation, Price, and Market Event scores
    res_final = res3.withColumn(
        "CONFIDENCE_SCORE",
        (col("VALIDATION_SCORE") * 0.6)
        + (col("PRICE_SCORE") * 0.2)
        + (col("MARKET_EVENT_SCORE") * 0.2),
    )

    # Determine the Final Confidence based on the Final Score
    res_final = res_final.withColumn(
        "CONFIDENCE_CATEGORY",
        when(col("CONFIDENCE_SCORE") > 0.4, "High").otherwise("Low"),
    )
    child_logger.info("Foreast Validation Done")
    return res_final
