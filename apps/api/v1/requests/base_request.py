"""
Base request validation class - Laravel-style
"""
from rest_framework.response import Response
from rest_framework import status
from typing import Dict, Any, Optional, List


class BaseRequest:
    """Base class for request validation"""

    def __init__(self, data: Dict[str, Any], files: Optional[Dict] = None):
        # Make a mutable dict copy of data if it's a QueryDict
        from django.http import QueryDict
        if isinstance(data, QueryDict):
            # Convert QueryDict to regular dict for full mutability
            self.data = dict(data.items())
        else:
            self.data = dict(data) if data else {}
        self.files = files or {}
        self.errors: Dict[str, List[str]] = {}

    def validate(self) -> bool:
        """Validate the request data"""
        self.errors = {}
        self.rules()
        return len(self.errors) == 0

    def rules(self):
        """Define validation rules - override in subclasses"""
        pass

    def validated(self) -> Dict[str, Any]:
        """Get validated data"""
        if not self.validate():
            raise ValidationException(self.errors)
        return self.data

    def fails(self) -> bool:
        """Check if validation fails"""
        return not self.validate()

    def errors_response(self) -> Response:
        """Return validation errors as Response"""
        return Response(
            {"errors": self.errors, "message": "Validation failed"},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    def _add_error(self, field: str, message: str):
        """Add validation error"""
        if field not in self.errors:
            self.errors[field] = []
        self.errors[field].append(message)

    def _required(self, field: str, field_name: str = None):
        """Check if field is required"""
        if field not in self.data or self.data[field] is None or self.data[field] == "":
            self._add_error(field, f"{field_name or field} is required")

    def _required_file(self, field: str, field_name: str = None):
        """Check if file is required"""
        if field not in self.files or not self.files[field]:
            self._add_error(field, f"{field_name or field} is required")

    def _integer(self, field: str, min_value: Optional[int] = None, max_value: Optional[int] = None):
        """Validate integer field"""
        if field in self.data:
            try:
                value = int(self.data[field])
                if min_value is not None and value < min_value:
                    self._add_error(field, f"{field} must be at least {min_value}")
                if max_value is not None and value > max_value:
                    self._add_error(field, f"{field} must be at most {max_value}")
                self.data[field] = value
            except (ValueError, TypeError):
                self._add_error(field, f"{field} must be an integer")

    def _float(self, field: str, min_value: Optional[float] = None, max_value: Optional[float] = None):
        """Validate float field"""
        if field in self.data:
            try:
                value = float(self.data[field])
                if min_value is not None and value < min_value:
                    self._add_error(field, f"{field} must be at least {min_value}")
                if max_value is not None and value > max_value:
                    self._add_error(field, f"{field} must be at most {max_value}")
                self.data[field] = value
            except (ValueError, TypeError):
                self._add_error(field, f"{field} must be a number")

    def _string(self, field: str, min_length: Optional[int] = None, max_length: Optional[int] = None):
        """Validate string field"""
        if field in self.data:
            value = str(self.data[field])
            if min_length is not None and len(value) < min_length:
                self._add_error(field, f"{field} must be at least {min_length} characters")
            if max_length is not None and len(value) > max_length:
                self._add_error(field, f"{field} must be at most {max_length} characters")
            self.data[field] = value

    def _boolean(self, field: str):
        """Validate boolean field"""
        if field in self.data:
            value = self.data[field]
            if isinstance(value, str):
                value = value.lower() in ("true", "1", "yes")
            elif not isinstance(value, bool):
                try:
                    value = bool(value)
                except:
                    self._add_error(field, f"{field} must be a boolean")
                    return
            self.data[field] = value

    def _url(self, field: str):
        """Validate URL field"""
        if field in self.data:
            value = str(self.data[field])
            if not (value.startswith("http://") or value.startswith("https://")):
                self._add_error(field, f"{field} must be a valid URL")

    def _in(self, field: str, allowed_values: List[Any]):
        """Validate field value is in allowed list"""
        if field in self.data and self.data[field] not in allowed_values:
            self._add_error(field, f"{field} must be one of: {', '.join(map(str, allowed_values))}")


class ValidationException(Exception):
    """Exception raised when validation fails"""

    def __init__(self, errors: Dict[str, List[str]]):
        self.errors = errors
        super().__init__("Validation failed")

