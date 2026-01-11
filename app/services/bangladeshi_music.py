"""
Bangladeshi music services for TheraMuse application.

Provides curated Bangladeshi artist queries and generational music context
for therapy recommendations based on patient demographics and preferences.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class BangladeshiSingerQueryGenerator:
    """
    Curated Bangladeshi artist query generator used for therapy recommendations.

    Generates personalized music search queries based on birthplace and
    favorite genre preferences using regional artist mappings.
    """

    REGION_KEYWORDS = {
        "dhaka": [
            "dhaka", "manikganj", "gopalganj", "faridpur", "munshiganj", "kushtia",
            "narayanganj", "gazipur"
        ],
        "chittagong": [
            "chittagong", "chattogram", "cox", "cox's bazar", "bandarban",
            "rangamati", "khagrachari"
        ],
        "sylhet": ["sylhet", "habiganj", "moulvibazar", "sunamganj"],
        "rajshahi": ["rajshahi", "naogaon", "chapainawabganj", "bogura", "pabna"],
        "rangpur": ["rangpur", "dinajpur", "kurigram", "gaibandha", "lalmonirhat"],
        "khulna": ["khulna", "jessore", "satkhira", "bagerhat", "narail", "jhenaidah"],
        "barisal": ["barisal", "barishal", "bhola", "jhalokathi", "pirojpur", "barguna"],
        "mymensingh": ["mymensingh", "sherpur", "jamalpur", "netrokona"],
    }

    REGION_ARTISTS = {
        "dhaka": [
            "Sabina Yasmin", "Runa Laila", "Feroza Begum", "Habib Wahid", "Arnob",
            "Tahsan Khan", "Momtaz Begum", "Andrew Kishore", "Subir Nandi",
            "Azam Khan", "Kumar Biswajit", "Shironamhin", "LRB", "Miles",
            "Nagar Baul", "Artcell", "Feedback"
        ],
        "chittagong": [
            "Ayub Bachchu", "Pritom Hasan", "Satya Saha", "Souls", "Feedback",
            "LRB", "Miles"
        ],
        "sylhet": [
            "Subir Nandi", "Runa Laila", "Shah Abdul Karim", "Hason Raja",
            "Rezwana Choudhury Bannya", "Shuvro Dev"
        ],
        "rajshahi": [
            "James", "Abdul Alim", "Abbasuddin Ahmed", "Warfaze", "Aurthohin"
        ],
        "rangpur": [
            "Abbasuddin Ahmed", "Rezwana Choudhury Bannya", "Momtaz Begum",
            "Farida Parvin"
        ],
        "khulna": [
            "Lalon Band", "Farida Parvin", "Fakir Alamgir", "Sharmin Sultana",
            "Sukumar Baidya"
        ],
        "barisal": [
            "Kumar Bishwajit", "Ferdous Wahid", "Shahnaz Rahmatullah",
            "Minar Rahman"
        ],
        "mymensingh": [
            "Momtaz Begum", "Imran Mahmudul", "Belal Khan", "Shahnaz Rahmatullah"
        ],
        "national": [
            "Sabina Yasmin", "Runa Laila", "Habib Wahid", "Arnb", "Tahsan Khan",
            "Andrew Kishore", "Subir Nandi", "Ayub Bachchu", "James", "Bappa Mazumder",
            "Momtaz Begum", "Abdul Jabbar", "Feroza Begum", "Azam Khan", "Hridoy Khan",
            "Asif Akbar", "Minar Rahman", "Imran Mahmudul", "Belal Khan", "Pritom Ahmed"
        ],
    }

    LOCATION_TEMPLATES = [
        "{location} nostalgic songs",
        "{location} folk songs",
        "{location} classic bangla songs",
        "{location} golden oldies",
        "{location} playback hits",
    ]

    def _normalize(self, value: Optional[str]) -> str:
        """Normalize string value for comparison."""
        return (value or "").strip().lower()

    def _match_region(self, birthplace: Optional[str]) -> str:
        """Match birthplace to region based on keywords."""
        normalized = self._normalize(birthplace)
        for region, keywords in self.REGION_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                return region
        return "national"

    def _build_artist_pool(self, region_key: str) -> List[str]:
        """Build artist pool for a given region."""
        pool = list(self.REGION_ARTISTS.get(region_key, []))
        if region_key != "national":
            pool += self.REGION_ARTISTS["national"][:10]
        if not pool:
            pool = list(self.REGION_ARTISTS["national"])
        return pool

    def _make_artist_query(self, artist: str, favorite_genre: str, variation: int) -> str:
        """Generate artist-specific query with variations."""
        genre = favorite_genre.strip().lower()
        if genre and variation % 3 == 0:
            return f"{artist} {genre} song"
        if variation % 5 == 0:
            return f"{artist} bengali folk song"
        if variation % 2 == 0:
            return f"{artist} bangla song"
        return f"{artist} bengali song"

    def get_queries(self, birthplace: Optional[str], favorite_genre: str = "") -> Dict:
        """
        Generate personalized music search queries.

        Args:
            birthplace: Patient's birthplace or location
            favorite_genre: Patient's preferred music genre

        Returns:
            Dictionary containing generated queries and metadata
        """
        location = (birthplace or "Bangladesh").strip()
        favorite_genre_clean = favorite_genre.strip()

        queries: List[str] = []

        if location:
            templates = list(self.LOCATION_TEMPLATES)
            if favorite_genre_clean:
                templates.append(f"{location} {favorite_genre_clean} songs")
                templates.append(f"{favorite_genre_clean} songs from {location}")
            for template in templates:
                if "{location}" in template:
                    queries.append(template.format(location=location))
                else:
                    queries.append(template)

        region_key = self._match_region(location)
        artist_pool = self._build_artist_pool(region_key)

        variation_index = 0
        while len(queries) < 24:
            artist = artist_pool[variation_index % len(artist_pool)]
            queries.append(self._make_artist_query(artist, favorite_genre_clean, variation_index))
            variation_index += 1

        deduped_queries: List[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = query.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped_queries.append(query)
            if len(deduped_queries) == 20:
                break

        return {
            "birthplace": location,
            "favorite_genre": favorite_genre_clean,
            "queries": deduped_queries[:20],
            "source": "curated_artist_database",
            "region": region_key,
        }


class BangladeshiGenerationalMatrix:
    """
    Provides generational music context for Bangladeshi patients.

    Maps birth years to musical eras and therapeutic ragas for
    personalized music therapy recommendations.
    """

    GENERATIONAL_RAGA_MAPPING = {
        (1931, 1955): {
            "musical_context": "Rabindra Sangeet, Nazrul Geeti, Baul, early film scores",
            "therapeutic_ragas": ["Yaman", "Bageshri", "Desh", "Khamaj", "Bhairavi"],
            "focus": "calmness, patriotic connection, spiritual grounding, rest, healing"
        },
        (1956, 1965): {
            "musical_context": "Band Revolution, Folk-Pop Fusion",
            "therapeutic_ragas": ["Kafi", "Pahadi", "Bhairavi"],
            "focus": "peace, tranquility, holistic healing, emotional balance"
        },
        (1966, 1980): {
            "musical_context": "Rock, Electro-Fusion, Indie Pop",
            "therapeutic_ragas": ["Darbari Kanada", "Durga", "Jogiya", "Maand"],
            "focus": "emotional balance, stability, focus, resilience, stress management"
        },
        (1981, 1995): {
            "musical_context": "Hip-Hop, EDM, Global Fusion",
            "therapeutic_ragas": ["Keeravani", "Charukeshi", "Gauri", "Hamsadhwani"],
            "focus": "upliftment, attention, optimism, relaxation, mental fatigue"
        }
    }

    def get_generational_context(self, birth_year: int) -> Dict:
        """
        Get generational musical context for a given birth year.

        Args:
            birth_year: Patient's birth year

        Returns:
            Dictionary containing generational context and therapeutic information
        """
        for (start_year, end_year), context in self.GENERATIONAL_RAGA_MAPPING.items():
            if start_year <= birth_year <= end_year:
                return {
                    "birth_year": birth_year,
                    "age_group": f"Born {start_year}-{end_year}",
                    **context
                }

        return {
            "birth_year": birth_year,
            "age_group": "Unknown",
            "musical_context": "General therapeutic music",
            "therapeutic_ragas": ["Yaman", "Bageshri", "Desh"],
            "focus": "general wellness"
        }

    def calculate_nostalgia_window(self, birth_year: int) -> Tuple[int, int]:
        """
        Calculate optimal nostalgia window for music therapy.

        Args:
            birth_year: Patient's birth year

        Returns:
            Tuple of (start_year, end_year) for nostalgic music preferences
        """
        return (birth_year + 10, birth_year + 30)