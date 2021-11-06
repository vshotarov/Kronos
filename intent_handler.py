from datetime import datetime, timedelta
from word2number import w2n
import re
import requests
import urllib.parse
import json


HOME_LOCATION_WOEID = 44418 # London

def get_location_info(location):
    request = requests.get("https://www.metaweather.com/api/location/search/?%s" %\
            urllib.parse.urlencode({"query":location}))
    return json.loads(request.text)

def interpret_time_delta(text):
    # This is going to be very crude
    # NOTE: Is it possible to train a neural net to convert a time slot
    # into usable data?
    text = text.strip()
    if "today" in text:
        return timedelta(seconds=max(
            (datetime.now().replace(hour=12,minute=0) - datetime.now()).total_seconds(), 0))
    elif "tomorrow" in text:
        return datetime.now().replace(hour=12) + timedelta(days=1)

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

    count = w2n.word_to_num(count)
    period = period.strip()
    period = period if period.endswith("s") else period + "s"

    return timedelta(**{period:count})

class IntentHandler(object):
    @staticmethod
    def weather_current(slots):
        if "location" not in slots.keys():
            location_woeid = HOME_LOCATION_WOEID
        else:
            location_info = get_location_info(slots["location"])
            if not location_info:
                print("Error in IntentHandler.weather_current: "
                      "Unrecognized location %s" % slots["location"])
                return

            location_woeid = location_info[0]["woeid"]

        request = requests.get("https://www.metaweather.com/api/location/%i" % location_woeid)
        weather_info = json.loads(request.text)

        first_forecast = weather_info["consolidated_weather"][0]

        print("The temperature in %s is currently %i degrees" % (
            weather_info["title"], first_forecast["the_temp"]))

    @staticmethod
    def weather_future(slots):
        if not "time" in slots.keys():
            print("Error in IntentHandler.weather_future: No time slot was provided")
            return

        print("get weather in", interpret_time_delta(slots["time"]))

    @staticmethod
    def time_current(slots):
        pass

    @staticmethod
    def time_timer(slots):
        pass

    @staticmethod
    def other(slots):
        pass

if __name__ == "__main__":
    IntentHandler.weather_current({})
