import pandas as pd
from geopy.distance import geodesic
import googlemaps

class GoogleMapsAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = googlemaps.Client(key=api_key)
    
    def get_distance(self, origin: tuple, destination: tuple, crow: bool = True) -> tuple:
        """
        Get the distance between two coordinates

        Arguments:
        ----------
        origin: tuple
            The coordinate of the origin
        destination: tuple
            The coordinate of the destination
        crow: bool
            If True, the distance is calculated as the crow flies. If False, the distance is calculated by road.

        Returns:
        --------
        tuple
            distance, duration
        """
        if crow:       
            distance = geodesic(origin, destination).meters
        else:
            try:
                distance = self.client.distance_matrix(origin, destination, mode='driving')['rows'][0]['elements'][0]['distance']['value']
            except KeyError:
                print(self.client.distance_matrix(origin, destination, mode='driving'))
        return distance
    
    def get_duration(self, origin: tuple, destination: tuple) -> int:
        """
        Get the duration between two coordinates

        Arguments:
        ----------
        origin: tuple
            The coordinate of the origin
        destination: tuple
            The coordinate of the destination

        Returns:
        --------
        int
            duration
        """
        duration = self.client.distance_matrix(origin, destination, mode='driving')['rows'][0]['elements'][0]['duration']['value']
        return duration