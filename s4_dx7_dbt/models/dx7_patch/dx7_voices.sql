
WITH cart_voices as (
    SELECT path, unnest(extract_voices(bytes)) voice from {{ ref("dx7_cartridges") }}
)
select path, voice.* from cart_voices