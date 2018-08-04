# v0.4.0

- update to `0.7` and drop `0.6` support

# v0.3.0

- Allow `NativeLavels` to specify a fallback label function to be used if
  an label that is not known to the encoder is encountered.

# v0.2.0

- drop `0.6.0-pre` and update syntax to `0.6`

- bug fix: make sure `convertlabel(LabelEnc.Indices, x, enc)`
  works for a single observation `x`

# v0.1.4

- support equality between `NativeLabels` for which `.labels` is equal
