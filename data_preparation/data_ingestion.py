# Databricks notebook source
import dlt

# Ingest bookings data
@dlt.table
def bookings():
    return (
        spark.read.format("csv")
        .option("header", True)
        .option("inferSchema", True)
        .load("/FileStore/booking_dummy_data.csv")
    )


# Ingest Flights data
@dlt.table()
def flights():
    return (
        spark.read.format("csv")
        .option("header", True)
        .option("inferSchema", True)
        .load("/FileStore/flight_dummy_data.csv")
    )


# Merge Bookings and flights into one wide table
@dlt.table()
def bookings_with_flights():
    bookings = dlt.read("bookings")
    flights = dlt.read("flights")
    return bookings.join(flights, ["flight_id"], how="inner")
