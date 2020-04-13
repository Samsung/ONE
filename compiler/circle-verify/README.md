# circle-verify

_circle-verify_ allows users to verify Circle models.

## Usage

Provide _circle_ file as a parameter to verify validity.

```
$ circle-verify circlefile.circle
```

Result for valid file
```
[ RUN       ] Check circlefile.circle
[      PASS ] Check circlefile.circle
```

Result for invalid file
```
[ RUN       ] Check circlefile.circle
[      FAIL ] Check circlefile.circle
```
