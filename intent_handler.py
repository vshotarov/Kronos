from datetime import datetime, timedelta
from word2number import w2n
from num2words import num2words
import re
import requests
import urllib.parse
import json
from timezonefinder import TimezoneFinder
import pytz


HOME_LOCATION_WOEID = 44418 # London
HOME_LOCATION_LATLONG = 51, 0

timezone_finder = TimezoneFinder()

def get_location_latlong(slots):
    if "location" not in slots.keys():
        return HOME_LOCATION_LATLONG, "London"
    else:
        # Bit weird to use a different api for this, but the metaweather one
        # seems to be a one man operation and i don't want to use it if i can help it
        request = requests.get("https://nominatim.openstreetmap.org/search?%s&format=json" % (
            urllib.parse.urlencode({"q":slots["location"]})))

        location_info = json.loads(request.text)

        if location_info:
            return (float(location_info[0]["lat"]), float(location_info[0]["lon"])),\
                location_info[0]["display_name"].split(",")[0]

        print("Error in intent_handler.get_location_latlong: "
              "Unrecognized location %s" % slots["location"])
        return None, None

def get_timezone_from_latlong(latlong):
    return timezone_finder.timezone_at(lat=latlong[0], lng=latlong[1])

def get_location_info(location):
    request = requests.get("https://www.metaweather.com/api/location/search/?%s" %\
            urllib.parse.urlencode({"query":location}))
    return json.loads(request.text)

def get_location_woeid(slots):
    if "location" not in slots.keys():
        return HOME_LOCATION_WOEID, "London"
    else:
        location_info = get_location_info(slots["location"])
        if not location_info:
            print("Error in intent_handler.get_location_woeid: "
                  "Unrecognized location %s" % slots["location"])
            return None, None

        return location_info[0]["woeid"], location_info[0]["title"]

def interpret_time_delta(text):
    # This is going to be very crude
    # NOTE: Is it possible to train a neural net to convert a time slot
    # into usable data?
    text = text.strip()
    if "today" in text:
        return timedelta(seconds=max(
            (datetime.now().replace(hour=12,minute=0) - datetime.now()).total_seconds(), 0))
    elif "tomorrow" in text:
        return (datetime.now().replace(hour=12) + timedelta(days=1)) - datetime.now()

    if text.startswith("in "):
        text = text[3:]
    elif text.startswith("a "):
        text = "one " + text[2:]
    elif text.startswith("an "):
        text = "one " + text[3:]

    match = re.match(
        "(.*?)(seconds?|minutes?|hours?|days?|weeks?|months?|years?)",
        text)

    if not match:
        return

    groups = match.groups()
    if len(groups) == 1:
        period = groups[0]
    elif len(groups) == 2:
        count, period = groups
    else:
        # If we've matched 0 or more than 2 groups, then we're probably
        # not dealing with a time delta, so we return nothing
        return

    count = count.strip()
    if not count:
        return

    try:
        count = w2n.word_to_num(count)
    except ValueError as e:
        print("Error in intent_handler.interpret_time_delta: Unrecognized number %s" % count)
        return

    period = period.strip()
    period = period if period.endswith("s") else period + "s"

    return timedelta(**{period:count})

class IntentHandler(object):
    @staticmethod
    def weather_current(slots):
        location_woeid, location_name = get_location_woeid(slots)

        if not location_woeid:
            return

        request = requests.get("https://www.metaweather.com/api/location/%i" % location_woeid)
        weather_info = json.loads(request.text)

        if not weather_info:
            print("Error in IntentHandler.weather_current: Could not find a forecast. slots - ", slots)

        first_forecast = weather_info["consolidated_weather"][0]

        return "The temperature in %s is currently %i degrees" % (
            location_name, first_forecast["the_temp"])

    @staticmethod
    def weather_future(slots):
        if not "time" in slots.keys():
            print("Error in IntentHandler.weather_future: No time slot was provided")
            return
        
        future_time_delta = interpret_time_delta(slots["time"])

        if not future_time_delta:
            print("Error in IntentHandler.weather_future: "
                "Could not interpret time %s" % slots["time"])
            return

        time_in_future = datetime.now() + future_time_delta

        location_woeid, location_name = get_location_woeid(slots)

        if not location_woeid:
            return

        request = requests.get("https://www.metaweather.com/api/location/%i/%s" % (
            location_woeid, time_in_future.strftime("%Y/%m/%d")))
        weather_info = json.loads(request.text)

        first_forecast = weather_info[0]

        return "The temperature in %s on the %s of %s will be %i degrees" % (
            location_name, num2words(time_in_future.day, ordinal=True),
            time_in_future.strftime("%B"), first_forecast["the_temp"])

    @staticmethod
    def time_current(slots):
        latlong, location_name = get_location_latlong(slots)

        if not latlong:
            return

        timezone = pytz.timezone(get_timezone_from_latlong(latlong))

        return "The current time in %s is %s" % (location_name,
                    datetime.now(timezone).strftime("%H:%M"))

    @staticmethod
    def time_timer(slots):
        pass

    @staticmethod
    def other(slots):
        print("Unrecognized command")
