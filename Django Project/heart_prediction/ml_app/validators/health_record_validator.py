from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

def validate_age(value):
    if value <= 20 :
        raise ValidationError(
            _('%(value)s is less than 20'),
            params={'value': value},
        )
    elif value >=100:
        raise ValidationError(
            _('%(value)s is over 100 '),
            params={'value': value},
        )

def validate_trebtps(value):
    if value <= 50 :
        raise ValidationError(
            _('%(value)s is less than 50'),
            params={'value': value},
        )
    elif value >=300:
        raise ValidationError(
            _('%(value)s is over 300 '),
            params={'value': value},
        )
def validate_thalach(value):
    if value <= 100 :
        raise ValidationError(
            _('%(value)s is less than 100'),
            params={'value': value},
        )
    elif value >=300:
        raise ValidationError(
            _('%(value)s is over 250 '),
            params={'value': value},
        )
def validate_oldpeak(value):
    if value < 0 :
        raise ValidationError(
            _('%(value)s is less than 0'),
            params={'value': value},
        )
    elif value >7:
        raise ValidationError(
            _('%(value)s is over 7 '),
            params={'value': value},
        )