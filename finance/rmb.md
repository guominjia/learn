# RMB

Press **Alt+F11** to open VB editor, paste below code, then save to `.xlsm` file, then you will have `RMB` function
```vb
Function RMB(num As Double) As String
    Dim digits As String
    ' 0-9 Chinese uppercase digits
    digits = ChrW(38646) & ChrW(22777) & ChrW(36144) & ChrW(21441) & _
             ChrW(32902) & ChrW(20237) & ChrW(38470) & ChrW(26578) & _
             ChrW(25420) & ChrW(29590)

    Dim u1 As String: u1 = ""
    Dim u2 As String: u2 = ChrW(25342)  ' shi (10)
    Dim u3 As String: u3 = ChrW(20336)  ' bai (100)
    Dim u4 As String: u4 = ChrW(20191)  ' qian (1000)
    Dim wan   As String: wan   = ChrW(19975)  ' wan (10,000)
    Dim yi    As String: yi    = ChrW(20159)  ' yi (100,000,000)
    Dim yuan  As String: yuan  = ChrW(20803)  ' yuan
    Dim jiaoC As String: jiaoC = ChrW(35282)  ' jiao (0.1)
    Dim fenC  As String: fenC  = ChrW(20998)  ' fen (0.01)
    Dim zheng As String: zheng = ChrW(25972)  ' zheng (exact)
    Dim ling  As String: ling  = ChrW(38646)  ' zero
    Dim fu    As String: fu    = ChrW(36127)  ' negative

    ' Group unit suffixes: (none), wan, yi, wan-yi
    Dim groupU(3) As String
    groupU(0) = ""
    groupU(1) = wan
    groupU(2) = yi
    groupU(3) = wan & yi

    Dim units(3) As String
    units(0) = u1: units(1) = u2: units(2) = u3: units(3) = u4

    ' Convert to fen (0.01) as Currency to avoid floating point errors
    Dim n As Currency
    n = CCur(Abs(num) * 100 + 0.5)

    ' Extract fen and jiao digits using Currency arithmetic to avoid Long overflow
    Dim fenV  As Integer: fenV  = CInt(n - Int(n / 10) * 10)
    Dim jiaoV As Integer: jiaoV = CInt(Int(n / 10) - Int(n / 100) * 10)
    Dim intPart As Currency: intPart = Int(n / 100)

    Dim intStr As String: intStr = ""
    Dim temp As Currency: temp = intPart
    Dim gi As Integer: gi = 0

    ' Process integer part in groups of 4 digits
    Do While temp > 0
        ' Use Currency modulo to avoid CLng overflow (Long max ~2.1 billion)
        Dim gLng As Long: gLng = CLng(temp - Int(temp / 10000) * 10000)
        Dim gStr As String: gStr = ""
        Dim i As Integer
        For i = 3 To 0 Step -1
            Dim d As Integer: d = (gLng \ (10 ^ i)) Mod 10
            If d <> 0 Then
                gStr = gStr & Mid(digits, d + 1, 1) & units(i)
            ElseIf Len(gStr) > 0 And Right(gStr, 1) <> ling Then
                gStr = gStr & ling
            End If
        Next i
        If Right(gStr, 1) = ling Then gStr = Left(gStr, Len(gStr) - 1)
        If Len(gStr) > 0 Then
            ' Insert zero when group is less than 1000 (missing leading digit)
            If Len(intStr) > 0 And gLng < 1000 Then
                intStr = gStr & groupU(gi) & ling & intStr
            Else
                intStr = gStr & groupU(gi) & intStr
            End If
        ElseIf Len(intStr) > 0 And Right(intStr, 1) <> ling Then
            intStr = ling & intStr
        End If
        temp = Int(temp / 10000)
        gi = gi + 1
    Loop

    If Len(intStr) > 0 Then
        If Right(intStr, 1) = ling Then intStr = Left(intStr, Len(intStr) - 1)
        intStr = intStr & yuan
    End If

    ' Build decimal part
    Dim decStr As String
    If jiaoV = 0 And fenV = 0 Then
        decStr = zheng
    ElseIf jiaoV = 0 Then
        decStr = ling & Mid(digits, fenV + 1, 1) & fenC
    ElseIf fenV = 0 Then
        decStr = Mid(digits, jiaoV + 1, 1) & jiaoC & zheng
    Else
        decStr = Mid(digits, jiaoV + 1, 1) & jiaoC & Mid(digits, fenV + 1, 1) & fenC
    End If

    If num < 0 Then intStr = fu & intStr
    If Len(intStr) = 0 Then intStr = ling & yuan
    RMB = intStr & decStr
End Function
```