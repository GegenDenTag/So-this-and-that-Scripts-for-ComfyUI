import numpy as np
from PIL import Image
import os
from typing import Optional, Tuple, Dict

class ColorTemperatureAnalyzer:
    """
    Klasse zur Analyse der Farbtemperatur von Bildern mit optionalem Weißabgleich
    """
    
    # Unterstützte Bildformate
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    
    def __init__(self):
        """
        Initialisiert den ColorTemperatureAnalyzer
        """
        self.image_path = None
        self.image_array = None
        self.white_reference_path = None
        self.white_reference_rgb = None
        self.white_balance_factors = None
        self.avg_rgb = None
        self.corrected_rgb = None
        self.color_temperature = None
        
    def load_white_reference(self, white_reference_path: str) -> bool:
        """
        Lädt eine Weißreferenz-Datei für den Weißabgleich        
        Args:
            white_reference_path (str): Pfad zur Weißreferenz-Datei            
        Returns:
            bool: True | False
        """
        try:
            if not os.path.exists(white_reference_path):
                print(f"Fehler: Weißreferenz-Datei '{white_reference_path}' nicht gefunden.")
                return False
            
            # Prüfen ob Format unterstützt wird
            file_extension = os.path.splitext(white_reference_path)[1].lower()
            if file_extension not in self.SUPPORTED_FORMATS:
                print(f"Fehler: Format '{file_extension}' wird nicht unterstützt.")
                return False
            
            # Weißreferenz-Bild laden
            white_img = Image.open(white_reference_path)
            
            # Zu RGB konvertieren falls nötig
            if white_img.mode != 'RGB':
                white_img = white_img.convert('RGB')
            
            white_array = np.array(white_img)
            
            # Durchschnittliche RGB-Werte der Weißreferenz berechnen
            self.white_reference_rgb = np.mean(white_array.reshape(-1, 3), axis=0) / 255.0
            
            # Weißabgleich-Faktoren berechnen
            # Faktoren um die Weißreferenz zu echtem Weiß (1.0, 1.0, 1.0) zu korrigieren
            self.white_balance_factors = 1.0 / self.white_reference_rgb
            
            # Normalisierung auf den niedrigsten Faktor (um Überbelichtung zu vermeiden)
            min_factor = np.min(self.white_balance_factors)
            self.white_balance_factors = self.white_balance_factors / min_factor
            
            self.white_reference_path = white_reference_path
            
            print(f"Weißreferenz erfolgreich geladen: {white_reference_path}")
            print(f"Weißreferenz RGB: R={self.white_reference_rgb[0]:.3f}, G={self.white_reference_rgb[1]:.3f}, B={self.white_reference_rgb[2]:.3f}")
            print(f"Weißabgleich-Faktoren: R={self.white_balance_factors[0]:.3f}, G={self.white_balance_factors[1]:.3f}, B={self.white_balance_factors[2]:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Fehler beim Laden der Weißreferenz: {e}")
            return False
    
    def create_white_reference_image(self, output_path: str, size: Tuple[int, int] = (100, 100)) -> bool:
        """
        Erstellt eine reine weiße Referenzdatei (#FFFFFF)        
        Args:
            output_path (str): Pfad für die zu erstellende Weißreferenz-Datei
            size (Tuple[int, int]): Größe des Bildes (Breite, Höhe)            
        Returns:
            bool: True | False
        """
        try:
            # Create
            white_image = Image.new('RGB', size, (255, 255, 255))
            white_image.save(output_path)
            
            #print(f"Weißreferenz-Bild erstellt: {output_path}")
            #print(f"Größe: {size}")
            
            return True
            
        except Exception as e:
            print(f"Fehler beim Erstellen der Weißreferenz: {e}")
            return False
    
    def load_image(self, image_path: str) -> bool:
        """
        Args:
            image_path (str): Pfad zum Bild            
        Returns:
            bool: True | False
        """
        try:
            # Prüfen ob Datei existiert
            if not os.path.exists(image_path):
                print(f"Fehler: Datei '{image_path}' nicht gefunden.")
                return False
            
            # Prüfen ob Format unterstützt wird
            file_extension = os.path.splitext(image_path)[1].lower()
            if file_extension not in self.SUPPORTED_FORMATS:
                print(f"Fehler: Format '{file_extension}' wird nicht unterstützt.")
                print(f"Unterstützte Formate: {', '.join(self.SUPPORTED_FORMATS)}")
                return False
            
            # laden
            img = Image.open(image_path)
            
            # Zu RGB konvertieren falls nötig
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            self.image_path = image_path
            self.image_array = np.array(img)
            
            print(f"Bild erfolgreich geladen: {image_path}")
            print(f"Bildgröße: {img.size}")
            print(f"Bildmodus: {img.mode}")
            
            return True
            
        except Exception as e:
            print(f"Fehler beim Laden des Bildes: {e}")
            return False
    
    def apply_white_balance(self, rgb_values: np.ndarray) -> np.ndarray:
        """
        Wendet den Weißabgleich auf RGB-Werte an        
        Args:
            rgb_values: RGB-Werte (0-1 Range)            
        Returns:
            np.ndarray: Weißabgleich-korrigierte RGB-Werte
        """
        if self.white_balance_factors is None:
            print("Warnung: Kein Weißabgleich geladen. Verwende ursprüngliche Werte.")
            return rgb_values
        
        # anwenden
        corrected = rgb_values * self.white_balance_factors
        
        # Auf 0-1 Range begrenzen
        corrected = np.clip(corrected, 0.0, 1.0)
        
        return corrected
    
    def calculate_average_rgb(self, apply_white_balance: bool = True) -> Optional[Tuple[float, float, float]]:
        """
        Berechnet die durchschnittlichen RGB-Werte des Bildes        
        Args:
            apply_white_balance (bool): Ob Weißabgleich angewendet werden soll            
        Returns:
            Tuple[float, float, float]: Durchschnittliche RGB-Werte (0-1)
        """
        if self.image_array is None:
            print("Fehler: Kein Bild geladen.")
            return None
        
        # RGB zu 0-1 Range normalisieren
        rgb_normalized = self.image_array / 255.0
        
        # Durchschnittliche Farbe berechnen
        self.avg_rgb = np.mean(rgb_normalized.reshape(-1, 3), axis=0)
        
        # Weißabgleich anwenden falls gewünscht und verfügbar
        if apply_white_balance and self.white_balance_factors is not None:
            self.corrected_rgb = self.apply_white_balance(self.avg_rgb)
            return tuple(self.corrected_rgb)
        else:
            self.corrected_rgb = self.avg_rgb
            return tuple(self.avg_rgb)
    
    def rgb_to_xyz(self, rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Konvertiert RGB zu XYZ Farbraum (sRGB Standard)        
        Args:
            rgb: RGB-Werte (0-1)            
        Returns:
            Tuple[float, float, float]: XYZ-Werte
        """
        r, g, b = rgb
        
        # Gamma-Korrektur (sRGB)
        def gamma_correct(value):
            if value <= 0.04045:
                return value / 12.92
            else:
                return pow((value + 0.055) / 1.055, 2.4)
        
        r = gamma_correct(r)
        g = gamma_correct(g)
        b = gamma_correct(b)
        
        # sRGB zu XYZ Matrix (D65 Illuminant)
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        return (x, y, z)
    
    def xyz_to_cct(self, xyz: Tuple[float, float, float]) -> Optional[float]:
        """
        Konvertiert XYZ zu korrelierter Farbtemperatur (CCT) in Kelvin
        Verwendet McCamy's Approximation        
        Args:
            xyz: XYZ-Werte            
        Returns:
            float: Farbtemperatur in Kelvin
        """
        x, y, z = xyz
        
        if x + y + z == 0:
            return None
        
        # XYZ zu Chromatizitätskoordinaten
        x_chrom = x / (x + y + z)
        y_chrom = y / (x + y + z)
        
        # McCamy's Approximation
        n = (x_chrom - 0.3320) / (0.1858 - y_chrom)
        cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
        
        return cct
    
    def analyze_color_temperature(self, apply_white_balance: bool = True) -> Optional[float]:
        """
        Analysiert die Farbtemperatur des geladenen Bildes        
        Args:
            apply_white_balance (bool): Ob Weißabgleich angewendet werden soll            
        Returns:
            float: Farbtemperatur in Kelvin
        """
        if self.image_array is None:
            print("Fehler: Kein Bild geladen.")
            return None
        
        # Durchschnittliche RGB-Werte berechnen
        avg_rgb = self.calculate_average_rgb(apply_white_balance)
        if avg_rgb is None:
            return None
        
        print(f"Ursprüngliche RGB-Werte: R={self.avg_rgb[0]:.3f}, G={self.avg_rgb[1]:.3f}, B={self.avg_rgb[2]:.3f}")
        
        if apply_white_balance and self.white_balance_factors is not None:
            print(f"Korrigierte RGB-Werte:   R={avg_rgb[0]:.3f}, G={avg_rgb[1]:.3f}, B={avg_rgb[2]:.3f}")
        
        # RGB zu XYZ konvertieren
        xyz = self.rgb_to_xyz(avg_rgb)
        print(f"XYZ-Werte: X={xyz[0]:.3f}, Y={xyz[1]:.3f}, Z={xyz[2]:.3f}")
        
        # XYZ zu Farbtemperatur
        self.color_temperature = self.xyz_to_cct(xyz)
        
        return self.color_temperature
    
    def get_physical_light_description(self, temp_kelvin: float) -> str:
        """
        Gibt eine Beschreibung der physikalischen Lichtquelle zurück        
        Args:
            temp_kelvin: Farbtemperatur in Kelvin            
        Returns:
            str: Beschreibung der physikalischen Lichtquelle
        """
        if temp_kelvin < 2000:
            return "Sehr warm - Kerzenlicht, Petroleum-/Gaslampe"
        elif temp_kelvin < 3000:
            return "Warm - Glühlampe, warmes LED-Licht"
        elif temp_kelvin < 4000:
            return "Warmweiß - Halogenlampe, warmweißes LED"
        elif temp_kelvin < 5000:
            return "Neutral - Leuchtstoffröhre, neutrales LED"
        elif temp_kelvin < 6000:
            return "Kaltweiß - Tageslicht bewölkt, kaltweißes LED"
        elif temp_kelvin < 7000:
            return "Tageslicht - direktes Sonnenlicht mittags"
        elif temp_kelvin < 8000:
            return "Kalt - Schatten im Freien, Nordfenster"
        else:
            return "Sehr kalt - klarer blauer Himmel, Hochgebirge"
    
    def get_camera_wb_setting_description(self, temp_kelvin: float) -> str:
        """
        Gibt die entsprechende Kamera/Photoshop Weißabgleich-Einstellung zurück        
        Args:
            temp_kelvin: Gemessene Farbtemperatur in Kelvin            
        Returns:
            str: Beschreibung der Kamera/Photoshop Einstellung die zu diesem Ergebnis führt
        """
        # Bei der Kamera/Photoshop Einstellung ist es umgekehrt!!!
        # Hohe Kelvin-Einstellung = warmes Bild (mehr Gelb/Rot)
        # Niedrige Kelvin-Einstellung = kaltes Bild (mehr Blau)
        
        if temp_kelvin < 2000:
            return "Kamera WB: ~8000-10000K+ (Maximum Warmton-Korrektur)"
        elif temp_kelvin < 3000:
            return "Kamera WB: ~6000-8000K (Starke Warmton-Korrektur)"
        elif temp_kelvin < 4000:
            return "Kamera WB: ~5000-6000K (Moderate Warmton-Korrektur)"
        elif temp_kelvin < 5000:
            return "Kamera WB: ~4000-5000K (Leichte Warmton-Korrektur)"
        elif temp_kelvin < 6000:
            return "Kamera WB: ~3500-4000K (Nahezu neutral)"
        elif temp_kelvin < 7000:
            return "Kamera WB: ~3000-3500K (Leichte Kaltton-Korrektur)"
        elif temp_kelvin < 8000:
            return "Kamera WB: ~2500-3000K (Moderate Kaltton-Korrektur)"
        else:
            return "Kamera WB: ~2000-2500K (Starke Kaltton-Korrektur)"
    
    def calculate_inverse_wb_setting(self, measured_temp: float) -> float:
        """
        Berechnet die ungefähre Kamera/Photoshop Weißabgleich-Einstellung,
        die zu der gemessenen Farbtemperatur führen würde        
        Args:
            measured_temp: Gemessene Farbtemperatur des Bildes            
        Returns:
            float: Geschätzte Kamera/Photoshop WB-Einstellung in Kelvin
        """
        # Vereinfachte inverse Beziehung
        # Je wärmer das Bild (niedrige gemessene Temp), desto höher war die WB-Einstellung
        
        if measured_temp <= 2000:
            return 9000  # Sehr hohe WB-Einstellung für sehr warmes Bild
        elif measured_temp <= 3000:
            return 7000  # Hohe WB-Einstellung
        elif measured_temp <= 4000:
            return 5500  # Moderate WB-Einstellung
        elif measured_temp <= 5500:
            return 4000  # Neutrale bis leicht warme WB-Einstellung
        elif measured_temp <= 7000:
            return 3200  # Leicht kalte WB-Einstellung
        elif measured_temp <= 8500:
            return 2800  # Kalte WB-Einstellung
        else:
            return 2300  # Sehr kalte WB-Einstellung
    
    def get_analysis_results(self, include_white_balance_info: bool = True) -> Dict:
        """
        Gibt alle Analyseergebnisse als Dictionary zurück        
        Args:
            include_white_balance_info (bool): Ob Weißabgleich-Informationen enthalten sein sollen            
        Returns:
            Dict: Vollständige Analyseergebnisse
        """
        if self.color_temperature is None:
            return {"error": "Keine Analyse durchgeführt"}
        
        estimated_wb_setting = self.calculate_inverse_wb_setting(self.color_temperature)
        
        results = {
            "image_path": self.image_path,
            "original_rgb": {
                "r": float(self.avg_rgb[0]),
                "g": float(self.avg_rgb[1]),
                "b": float(self.avg_rgb[2])
            },
            "analyzed_rgb": {
                "r": float(self.corrected_rgb[0]),
                "g": float(self.corrected_rgb[1]),
                "b": float(self.corrected_rgb[2])
            },
            "color_temperature_analysis": {
                "measured_kelvin": round(self.color_temperature, 0),
                "physical_light_source": self.get_physical_light_description(self.color_temperature),
                "estimated_camera_wb_setting": round(estimated_wb_setting, 0),
                "camera_wb_description": self.get_camera_wb_setting_description(self.color_temperature)
            }
        }
        
        if include_white_balance_info and self.white_balance_factors is not None:
            results["white_balance"] = {
                "reference_path": self.white_reference_path,
                "reference_rgb": {
                    "r": float(self.white_reference_rgb[0]),
                    "g": float(self.white_reference_rgb[1]),
                    "b": float(self.white_reference_rgb[2])
                },
                "correction_factors": {
                    "r": float(self.white_balance_factors[0]),
                    "g": float(self.white_balance_factors[1]),
                    "b": float(self.white_balance_factors[2])
                }
            }
        
        return results
    
    def analyze_image_with_white_balance(self, image_path: str, white_reference_path: str) -> Optional[Dict]:
        """
            Komplette Analyse eines Bildes mit Weißabgleich        
        Args:
            image_path: Pfad zum zu analysierenden Bild
            white_reference_path: Pfad zur Weißreferenz-Datei            
        Returns:
            Dict: Analyseergebnisse oder None bei Fehler
        """
        if not self.load_white_reference(white_reference_path):
            return None
            
        if not self.load_image(image_path):
            return None
        
        temp = self.analyze_color_temperature(apply_white_balance=True)
        if temp is None:
            return None
        
        return self.get_analysis_results()
    
    def analyze_image(self, image_path: str, apply_white_balance: bool = True) -> Optional[Dict]:
        """
        Komplette Analyse eines Bildes (Laden + Analysieren)        
        Args:
            image_path: Pfad zum Bild
            apply_white_balance: Ob ein bereits geladener Weißabgleich angewendet werden soll            
        Returns:
            Dict: Analyseergebnisse oder None bei Fehler
        """
        if not self.load_image(image_path):
            return None
        
        temp = self.analyze_color_temperature(apply_white_balance)
        if temp is None:
            return None
        
        return self.get_analysis_results()


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Analyzer erstellen
    analyzer = ColorTemperatureAnalyzer()
    
    # Beispiel-Pfade (anpassen!)
    image_path = "Path/to/file/warme-Lichtstimmung-8200-Kelvin.jpg"  # Zu analysierendes Bild
    white_reference_path = "Path/to/file/white_reference.png"  # Weißreferenz-Datei
    
    #print("=== Farbtemperatur-Analyse mit Weißabgleich ===")
    
    # Weißreferenz-Datei erstellen (falls noch nicht vorhanden)
    #print("\n1. Erstelle Weißreferenz-Datei...")
    if analyzer.create_white_reference_image(white_reference_path, (100, 100)):
        print("Weißreferenz-Datei erfolgreich erstellt!")
    
    # Komplette Analyse mit Weißabgleich
    #print(f"\n2. Analysiere Bild mit Weißabgleich...")
    #print(f"Bild: {image_path}")
    #print(f"Weißreferenz: {white_reference_path}")
    #print("-" * 50)
    
    results = analyzer.analyze_image_with_white_balance(image_path, white_reference_path)
    
    if results:
        print("\n=== ERGEBNISSE ===")
        print(f"Ursprüngliche RGB-Werte:")
        print(f"  R: {results['original_rgb']['r']:.3f}")
        print(f"  G: {results['original_rgb']['g']:.3f}")
        print(f"  B: {results['original_rgb']['b']:.3f}")
        
        print(f"\nKorrigierte RGB-Werte:")
        print(f"  R: {results['analyzed_rgb']['r']:.3f}")
        print(f"  G: {results['analyzed_rgb']['g']:.3f}")
        print(f"  B: {results['analyzed_rgb']['b']:.3f}")
        
        if 'white_balance' in results:
            print(f"\nWeißabgleich-Faktoren:")
            print(f"  R: {results['white_balance']['correction_factors']['r']:.3f}")
            print(f"  G: {results['white_balance']['correction_factors']['g']:.3f}")
            print(f"  B: {results['white_balance']['correction_factors']['b']:.3f}")
        
        print(f"\n=== FARBTEMPERATUR-ANALYSE ===")
        cta = results['color_temperature_analysis']
        print(f"Gemessene Farbtemperatur: {cta['measured_kelvin']:.0f} K")
        print(f"Physikalische Lichtquelle: {cta['physical_light_source']}")
        print(f"\nGeschätzte Kamera/Photoshop WB-Einstellung: {cta['estimated_camera_wb_setting']:.0f} K")
        print(f"Kamera WB-Beschreibung: {cta['camera_wb_description']}")
        
        #print(f"\n=== ERKLÄRUNG ===")
        #print(f"• Das Bild zeigt Licht mit ca. {cta['measured_kelvin']:.0f}K Farbtemperatur")
        #print(f"• Dies entspricht: {cta['physical_light_source']}")
        #print(f"• Um dieses Ergebnis zu erzielen, wurde vermutlich eine")
        #print(f"  Kamera/Photoshop WB-Einstellung von ca. {cta['estimated_camera_wb_setting']:.0f}K verwendet")
        #print(f"• {cta['camera_wb_description']}")
    else:
        print("Analyse fehlgeschlagen.")
    
    print("\n" + "="*60)
    
    # Vergleich ohne Weißabgleich
    print("\n3. Vergleich ohne Weißabgleich...")
    analyzer2 = ColorTemperatureAnalyzer()
    results_no_wb = analyzer2.analyze_image(image_path, apply_white_balance=False)
    
    if results_no_wb:
        cta_no_wb = results_no_wb['color_temperature_analysis']
        print(f"Ohne Weißabgleich: {cta_no_wb['measured_kelvin']:.0f} K ({cta_no_wb['physical_light_source']})")
        if results:
            cta_wb = results['color_temperature_analysis']
            print(f"Mit Weißabgleich:  {cta_wb['measured_kelvin']:.0f} K ({cta_wb['physical_light_source']})")
            diff = abs(cta_wb['measured_kelvin'] - cta_no_wb['measured_kelvin'])
            print(f"Differenz: {diff:.0f} K")
            
            print(f"\nKamera/Photoshop WB-Einstellungen:")
            print(f"Ohne WB-Korrektur: ~{cta_no_wb['estimated_camera_wb_setting']:.0f} K")
            print(f"Mit WB-Korrektur:  ~{cta_wb['estimated_camera_wb_setting']:.0f} K")
