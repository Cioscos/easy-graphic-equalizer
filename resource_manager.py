from pathlib import Path


class ResourceManager:
    """
    Utility class to get acces to the resources in the application
    """
    def __init__(self) -> None:
        self.base_path = Path(__file__).parent / "resources"
    
    def get_image_path(self, image_name: str, subfolder: str = "") -> Path:
        """
        Get path of the searched image

        Args:
            image_name (Path): the name of the image
            subfolder (str, optional): The name of the subfolder of the parent folder. Defaults to "".

        Raises:
            FileNotFoundError: if no file is found
        """
        path: Path = self.base_path / subfolder / image_name
        if path.exists() and path.is_file():
            return path
        raise FileNotFoundError(f"Image '{image_name}' not found in subfolder '{subfolder}'")
