const locationButton = document.getElementById("show-location");

if (locationButton) {
    const locationStatus = document.getElementById("location-status");
    const mapFrame = document.getElementById("location-map-frame");
    const map = document.getElementById("location-map");

    locationButton.addEventListener("click", () => {
        if (!navigator.geolocation) {
            locationStatus.textContent = "Location is not available in this browser.";
            return;
        }

        locationButton.disabled = true;
        locationStatus.textContent = "Waiting for your browser's location permission...";

        navigator.geolocation.getCurrentPosition(
            ({ coords }) => {
                const coordinates = encodeURIComponent(`${coords.latitude},${coords.longitude}`);
                map.src = `https://maps.google.com/maps?q=${coordinates}&z=15&output=embed`;
                mapFrame.hidden = false;
                locationButton.disabled = false;
                locationStatus.textContent = "Google Maps is centered on the location you shared.";
            },
            (error) => {
                locationButton.disabled = false;
                locationStatus.textContent = error.code === error.PERMISSION_DENIED
                    ? "Location permission was denied. Allow access in your browser settings to view the map."
                    : "We couldn't get your location. Check your connection and try again.";
            },
            { enableHighAccuracy: true, maximumAge: 60000, timeout: 10000 },
        );
    });
}